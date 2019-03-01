from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Activation, BatchNormalization, Dropout


class DummyBackboneNetwork():
    def __init__(self, input_shape, batch_size
                , first_filter_size=128, depth=4, with_dropout_lid=None
                , class_num=21):
        self.__input_shape = input_shape
        self.__batch_size = batch_size
        self.__first_filter_size = first_filter_size
        self.__depth = depth
        self.__with_dropout_lid = with_dropout_lid
        if self.__with_dropout_lid is None:
            self.__with_dropout_lid = [3,4]
        self.__class_num = class_num

        inputs = Input(self.__input_shape)

        layer = inputs
        filter_size = self.__first_filter_size
        conv_layer = []

        for d in range(self.__depth):
            if d != 0:
                layer = MaxPooling2D()(layer)
            cv1 = Conv2D(filter_size, 3, padding='same')(layer)
            cv1_bn = BatchNormalization()(cv1)
            cv1_ac = Activation('relu')(cv1_bn)
            cv2 = Conv2D(filter_size, 3, padding='same')(cv1_ac)
            cv2_bn = BatchNormalization()(cv2)
            cv2_ac = Activation('relu')(cv2_bn)
            layer = cv2_ac
            if d in self.__with_dropout_lid:
                layer = Dropout(0.5)(layer)
            conv_layer.append(layer)
            filter_size = filter_size * 2

        flt = Flatten()(layer)
        fc1 = Dense(256)(flt)
        fc1_ac = Activation('relu')(fc1)
        fc2 = Dense(64)(fc1_ac)
        fc2_ac = Activation('relu')(fc2)
        fc3 = Dense(self.__class_num)(fc2_ac)
        fc3_ac = Activation('softmax')(fc3)
        outputs = fc3_ac

        self.__model = Model(inputs=[inputs], outputs=[outputs])

        self.input_layer = inputs
        self.intermediate_layer = conv_layer[self.__depth - 2]
        self.last_intermediate_layer = conv_layer[self.__depth - 1]
