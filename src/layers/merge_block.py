from keras.layers import Concatenate, Reshape, Activation
from .prior_box_layer import PriorBoxLayer

import keras.backend as KB

class MergeBlock():
    def __init__(self, class_num=21):
        self.__class_num = class_num


    def __call__(self, inputs):
        return self.__merge_block(inputs)


    def __merge_block(self, inputs):
        loc_layers = []
        conf_layers = []
        pbox_layers = []

        for loc, conf, pbox in inputs:
            loc_layers.append(loc)
            conf_layers.append(conf)
            pbox = PriorBoxLayer(pbox)(loc)
            pbox_layers.append(pbox)

        locs = Concatenate(axis=1)(loc_layers)
        _, box_num, _ = locs.get_shape().as_list()

        merge_conf = Concatenate(axis=1)(conf_layers)
        confs = Activation('softmax')(merge_conf)

        pboxs = Concatenate(axis=1, name='pbox_concat')(pbox_layers)

        merge_layer = Concatenate(axis=2)([locs, confs, pboxs])
        return merge_layer
