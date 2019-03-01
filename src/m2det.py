import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Input
from keras.optimizers import Adam, SGD
from keras.utils import multi_gpu_model, plot_model
import keras.backend as KB

from layers import RegionBlock, MergeBlock
from layers import FeatureFusionModuleV1, FeatureFusionModuleV2
from layers import ThinnedUShapeModule, ScalewiseFeatureAggregationModule
from multibox_loss import MultiboxLoss
from dummy_backbone_network import DummyBackboneNetwork


class M2Det(object):
    def __init__(self, input_shape, batch_size, class_num=21
                    , backbone_inputs=None
                    , backbone_net_layer_1=None, backbone_net_layer_2=None
                    , lebels=8, scales=6):
        ### set value
        self.__input_shape = input_shape
        self.__batch_size = batch_size
        self.__class_num = class_num
        self.__backbone_inputs = backbone_inputs
        self.__backbone_net_layer_1 = backbone_net_layer_1
        self.__backbone_net_layer_2 = backbone_net_layer_2
        self.__tum_depth = lebels
        self.__tum_encode_depth = scales - 1
        self.__sfam_scales = scales

        ### create network
        if self.__backbone_inputs is None or self.__backbone_net_layer_1 is None or self.__backbone_net_layer_2 is None :
            backbone_net = DummyBackboneNetwork(self.__input_shape, self.__batch_size)
            self.__backbone_inputs = backbone_net.input_layer
            self.__backbone_net_layer_1 = backbone_net.intermediate_layer
            self.__backbone_net_layer_2 = backbone_net.last_intermediate_layer

        base_feature = FeatureFusionModuleV1()([self.__backbone_net_layer_1, self.__backbone_net_layer_2])

        layer = base_feature
        feature_pyramids = []
        for d in range(self.__tum_depth):
            if not (d == 0):
                layer = FeatureFusionModuleV2()([base_feature, layer])
            else:
                # tmp
                layer = Conv2D(256, 1, padding='same', activation='relu')(layer)

            feature_pyramid = ThinnedUShapeModule(depth=self.__tum_encode_depth)(layer)
            feature_pyramids.append(feature_pyramid)
            layer = feature_pyramid[self.__tum_encode_depth]

        ml_feature_pyramid = ScalewiseFeatureAggregationModule(scales=self.__sfam_scales)(feature_pyramids)

        prediction_layers = []
        for k, ml_feature in enumerate(ml_feature_pyramid):
            # TODO : refactoring
            if k == 0:
                region = RegionBlock(class_num, batch_size, input_shape, 30
                                    , aspect_ratios=[2], priors=3, normalize_scale=20)(ml_feature)
            else:
                max_size = 60 + 54 * k
                region = RegionBlock(class_num, batch_size, input_shape, max_size - 54, max_size=max_size)(ml_feature)
            prediction_layers.append(region)

        outputs = MergeBlock(class_num=class_num)(prediction_layers)

        self.__model = Model(inputs=[self.__backbone_inputs], outputs=[outputs])
        self.__set_prior_boxes([prior_box for _, _, prior_box in prediction_layers])


    def compile_model(self):
        self.__model.compile(optimizer=Adam(lr=0.00001), loss=MultiboxLoss(self.__class_num, self.__batch_size).loss)


    def get_model(self, with_compile=False):
        if with_compile:
            self.compile_model()
        return self.__model


    def get_parallel_model(self, gpu_num, with_compile=False):
        self.__model = multi_gpu_model(self.__model, gpus=gpu_num)
        return self.get_model(with_compile)


    def __set_prior_boxes(self, prior_box_list):
        self.__prior_boxes = np.concatenate(prior_box_list, axis=0)


    def get_prior_boxes(self):
        return self.__prior_boxes


    def show_model_summary(self):
        self.__model.summary()


    def plot_model_summary(self, file_name):
        plot_model(self.__model, to_file=file_name)

