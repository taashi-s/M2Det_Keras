from keras.layers import Dense, Concatenate
from keras.layers import GlobalAveragePooling2D, Multiply
from .squeeze_and_excitation_block import SqueezeAndExcitationBlock

class ScalewiseFeatureAggregationModule():
    def __init__(self, scales=6):
        self.__scales = scales


    def __call__(self, inputs):
        return self.__scalewise_feature_aggregation(inputs)


    def __scalewise_feature_aggregation(self, inputs):
        same_scale_layers_list = []
        for _ in range(self.__scales):
            same_scale_layers_list.append([])

        for feature_pyramid in inputs:
            for k, feature in enumerate(feature_pyramid[:self.__scales]):
                same_scale_layers_list[k].append(feature)

        multi_level_layer_feature_pyramid = []
        for same_scale_layers in same_scale_layers_list:
            multi_level_layer = Concatenate()(same_scale_layers)
            multi_level_layer_feature = SqueezeAndExcitationBlock()(multi_level_layer)

            multi_level_layer_feature_pyramid.append(multi_level_layer_feature)

        return multi_level_layer_feature_pyramid

