import numpy as np
import keras.backend as KB
from keras.engine.base_layer import Layer, InputSpec


class NormalizeLayer(Layer):
    def __init__(self, scale, **kwargs):
        super(NormalizeLayer, self).__init__(**kwargs)
        self.__scale = scale


    def build(self, input_shape):
        self.__input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[3],)
        init_gamma = self.__scale * np.ones(shape)
        self.__gamma = KB.variable(init_gamma, name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.__gamma]


    def call(self, x):
        output = KB.l2_normalize(x, 3)
        output *= self.__gamma
        return output


    def compute_output_shape(self, input_shape):
        return input_shape
