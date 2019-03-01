import keras.backend as KB
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Activation, Multiply

class SqueezeAndExcitationBlock():
    def __init__(self, ratio=16):
        self.__ratio = ratio


    def __call__(self, inputs):
        return self.__squeeze_and_excitation(inputs)


    def __squeeze_and_excitation(self, inputs):
        ch = inputs.get_shape().as_list()[3]
        gave = GlobalAveragePooling2D()(inputs)

        fc1 = Dense(ch//self.__ratio)(gave)
        fc1_ac = Activation('relu')(fc1)

        fc2 = Dense(ch)(fc1_ac)
        fc2_ac = Activation('sigmoid')(fc2)

        return Multiply()([inputs, fc2_ac])
