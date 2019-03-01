from keras.layers import Conv2D, Flatten, Dense, Reshape


class LocationLayer():
    def __init__(self, priors, use_dense=False):
        self.__priors = priors
        self.__use_dense = use_dense


    def __call__(self, inputs):
        if self.__use_dense:
            return self.__location_layer_dense(inputs)
        else:
            return self.__location_layer_conv(inputs)


    def __location_layer_dense(self, inputs):
        location = Dense(self.__priors * 4)(inputs)
        location_res = Reshape((-1, 4))(location)
        return location_res


    def __location_layer_conv(self, inputs):
        conv = Conv2D(self.__priors * 4, 3, padding='same')(inputs)
        location_res = Reshape((-1, 4))(conv)
        return location_res
