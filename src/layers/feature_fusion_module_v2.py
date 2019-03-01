from keras.layers import Conv2D, Concatenate


class FeatureFusionModuleV2():
    def __init__(self):
      pass


    def __call__(self, inputs):
        return self.__feature_fusion(inputs)


    def __feature_fusion(self, inputs):
        feature_layer = inputs[0]
        layer = inputs[1]

        cv1 = Conv2D(128, 1, padding='same')(feature_layer)

        layer = Concatenate()([cv1, layer])
        return layer

