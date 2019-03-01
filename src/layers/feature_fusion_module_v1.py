from keras.layers import Conv2D, UpSampling2D, Concatenate


class FeatureFusionModuleV1():
    def __init__(self):
      pass


    def __call__(self, inputs):
        return self.__feature_fusion(inputs)


    def __feature_fusion(self, inputs):
        feature_layer = inputs[0]
        last_feature_layer = inputs[1]

        cv1 = Conv2D(512, 1, padding='same')(last_feature_layer)
        up1 = UpSampling2D()(cv1)

        cv2 = Conv2D(256, 1, padding='same')(feature_layer)

        layer = Concatenate()([cv2, up1])
        return layer

