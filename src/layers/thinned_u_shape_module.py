from keras.layers import Conv2D, UpSampling2D, Add


class ThinnedUShapeModule():
    def __init__(self, depth=5, encode_filter=256, outputs_filter=128):
      self.__depth = depth
      self.__filter = encode_filter
      self.__outputs_filter = outputs_filter
      pass


    def __call__(self, inputs):
        return self.__thinned_u_shape(inputs)


    def __thinned_u_shape(self, inputs):
        layer = inputs

        encode_layers = []
        encode_layers.append(layer)
        for d in range(self.__depth):
          layer = Conv2D(self.__filter, 3, strides=2, padding='same')(layer)
          encode_layers.append(layer)


        encode_layers.reverse()

        feature_pyramid = []
        layer = encode_layers[0]
        feature_layer = Conv2D(self.__outputs_filter, 1, padding='same')(layer)
        feature_pyramid.append(feature_layer)
        for encode_layer in encode_layers[1:]:
            cv = Conv2D(self.__filter, 3, padding='same')(layer)
            up = UpSampling2D()(cv)
            layer = Add()([up, encode_layer])
            feature_layer = Conv2D(self.__outputs_filter, 1, padding='same')(layer)
            feature_pyramid.append(feature_layer)

        return feature_pyramid

