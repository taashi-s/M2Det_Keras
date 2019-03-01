import numpy as np
import keras.backend as KB
from keras.engine.base_layer import Layer


class PriorBoxLayer(Layer):
    def __init__(self, prior_box, **kwargs):
        self.__pb = prior_box
        super(PriorBoxLayer, self).__init__(**kwargs)


    def call(self, x):
        prior_boxes_3d = KB.expand_dims(KB.variable(self.__pb), 0)
        prior_boxes_tile = KB.tile(prior_boxes_3d, [KB.shape(x)[0], 1, 1])
        return prior_boxes_tile


    def compute_output_shape(self, input_shape):
        pb_shape = np.shape(self.__pb)
        return (None, pb_shape[0], pb_shape[1])

