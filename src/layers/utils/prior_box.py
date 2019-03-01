import numpy as np
import keras.backend as KB
from keras.engine.base_layer import Layer


class PriorBox():
    def __init__(self, img_size, min_size, max_size=None
                 , aspect_ratios=None, variances=[0.1]):
        self.__img_size = img_size
        self.__min = min_size
        self.__max = max_size
        self.__aspect_ratios = self.__calc_aspect_ratios(aspect_ratios)
        self.__variances = np.array(variances)


    def __calc_aspect_ratios(self, aspect_ratios):
        asp_ratios = [1.0]
        if isinstance(self.__max, int):
            asp_ratios.append(1.0)
        if isinstance(aspect_ratios, list):
            for ar in aspect_ratios:
                if ar in asp_ratios:
                    continue
                asp_ratios.append(ar)
                asp_ratios.append(1.0 / ar)
        return asp_ratios


    def __call__(self, h, w):
        box_heights, box_widths = self.__get_prior_box_shapes()
        centers_x, centers_y = self.__get_center_positions(h, w)
        prior_boxes = self.__get_prior_boxes(box_heights, box_widths, centers_x, centers_y)
        variances = self.__get_variances(len(prior_boxes))

        prior_boxes_cancat_variance = np.concatenate((prior_boxes, variances), axis=1)
        #prior_boxes_3d = KB.expand_dims(KB.variable(prior_boxes_cancat_variance), 0)
        #prior_boxes_3d = KB.tile(prior_boxes_3d, [batch_size, 1, 1])
        #return prior_boxes_3d
        return prior_boxes_cancat_variance


    def __get_prior_box_shapes(self):
        box_widths = []
        box_heights = []
        for ar in self.__aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.__min)
                box_heights.append(self.__min)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.__min * self.__max))
                box_heights.append(np.sqrt(self.__min * self.__max))
            elif ar != 1:
                box_widths.append(self.__min * np.sqrt(ar))
                box_heights.append(self.__min / np.sqrt(ar))
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)
        return box_heights, box_widths


    def __get_center_positions(self, h, w):
        img_h, img_w, _ = self.__img_size

        step_x = img_w / w
        step_y = img_h / h
        linx = np.linspace(0.5 * step_x, img_w - 0.5 * step_x, w)
        liny = np.linspace(0.5 * step_y, img_h - 0.5 * step_y, h)
        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)
        return centers_x, centers_y


    def __get_prior_boxes(self, box_heights, box_widths, centers_x, centers_y):
        img_h, img_w, _ = self.__img_size

        prior_num = len(self.__aspect_ratios)
        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * prior_num))
        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights
        prior_boxes[:, ::2] /= img_w
        prior_boxes[:, 1::2] /= img_h
        prior_boxes = prior_boxes.reshape(-1, 4)
        prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
        return prior_boxes


    def __get_variances(self, box_num):
        variances = np.array([])
        if len(self.__variances) == 1:
            variances = np.ones((box_num, 4)) * self.__variances[0]
        elif len(self.__variances) == 4:
            variances = np.tile(self.__variances, (box_num, 1))
        return variances


#    def compute_output_shape(self, input_shape):
#        prior_num = len(self.__aspect_ratios)
#        _, h, w, _ = input_shape
#        box_num = prior_num * h * w
#        return (None, box_num, 8)
