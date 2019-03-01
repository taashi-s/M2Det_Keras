import os
import numpy as np
from matplotlib import pyplot
import keras.callbacks as KC
import math
import pickle
import cv2
import random

from m2det import M2Det
from images_loader import load_images, save_images
from option_parser import get_option
from data_generator import DataGenerator
from history_checkpoint_callback import HistoryCheckpoint
from utils import BBoxUtility


CLASS_NUM = 1 + 1 # background + class
#INPUT_IMAGE_SHAPE = (320, 320, 3)
INPUT_IMAGE_SHAPE = (512, 512, 3)
BATCH_SIZE = 20
EPOCHS = 500
GPU_NUM = 8
WITH_NORM = True

DIR_BASE = os.path.join('.', '..')
DIR_MODEL = os.path.join(DIR_BASE, 'model')
DIR_TRAIN_INPUTS = os.path.join(DIR_BASE, 'inputs')
DIR_TRAIN_TEACHERS = os.path.join(DIR_BASE, 'teachers')
DIR_VALID_INPUTS = os.path.join(DIR_BASE, 'valid_inputs')
DIR_VALID_TEACHERS = os.path.join(DIR_BASE, 'valid_teachers')
DIR_OUTPUTS = os.path.join(DIR_BASE, 'outputs')
DIR_TEST = os.path.join(DIR_BASE, 'predict_data')
DIR_PREDICTS = os.path.join(DIR_BASE, 'predict_data')

FILE_MODEL = 'model.hdf5'


def train(gpu_num=None, with_generator=False, load_model=False, show_info=True):
    print('network creating ... ', end='', flush=True)
    network = M2Det(INPUT_IMAGE_SHAPE, BATCH_SIZE, class_num=CLASS_NUM)
    print('... created')

    if show_info:
        network.plot_model_summary('../model_plot.png')
        network.show_model_summary()
    if isinstance(gpu_num, int):
        model = network.get_parallel_model(gpu_num, with_compile=True)
    else:
        model = network.get_model(with_compile=True)

    model_filename = os.path.join(DIR_MODEL, FILE_MODEL)
    callbacks = [ KC.TensorBoard()
                , HistoryCheckpoint(filepath='LearningCurve_{history}.png'
                                    , verbose=1
                                    , period=10
                                   )
                , KC.ModelCheckpoint(filepath=model_filename
                                     , verbose=1
                                     , save_weights_only=True
                                     , save_best_only=True
                                     , period=10
                                    )
                ]

    if load_model:
        print('loading weghts ... ', end='', flush=True)
        model.load_weights(model_filename)
        print('... loaded')

    print('data generating ...', end='', flush=True)
    priors = network.get_prior_boxes()
    bbox_util = BBoxUtility(CLASS_NUM, priors)

    train_generator = DataGenerator(DIR_TRAIN_INPUTS, DIR_TRAIN_TEACHERS, bbox_util
                                    , INPUT_IMAGE_SHAPE, with_norm=WITH_NORM)
    valid_generator = DataGenerator(DIR_VALID_INPUTS, DIR_VALID_TEACHERS, bbox_util
                                    , INPUT_IMAGE_SHAPE, with_norm=WITH_NORM)
    print('... created')

    if with_generator:
        train_data_num = train_generator.data_size()
        valid_data_num = valid_generator.data_size()
        history = model.fit_generator(train_generator.generator(batch_size=BATCH_SIZE)
                                      , steps_per_epoch=math.ceil((train_data_num / BATCH_SIZE) * 2)
                                      , epochs=EPOCHS
                                      , verbose=1
                                      , use_multiprocessing=True
                                      , callbacks=callbacks
                                      , validation_data=valid_generator.generator(batch_size=BATCH_SIZE)
                                      , validation_steps=math.ceil(valid_data_num / BATCH_SIZE)
                                     )
    else:
        print('data generateing ... ') #, end='', flush=True)
        train_inputs, train_teachers = train_generator.generate_data(batch_size=BATCH_SIZE)
        valid_data = valid_generator.generate_data(batch_size=BATCH_SIZE)
        print('... generated')
        history = model.fit(train_inputs, train_teachers, batch_size=BATCH_SIZE, epochs=EPOCHS
                            , validation_data=valid_data
                            , shuffle=True, verbose=1, callbacks=callbacks)

    print('model saveing ... ', end='', flush=True)
    model.save_weights(model_filename)
    print('... saved')
    print('learning_curve saveing ... ', end='', flush=True)
    save_learning_curve(history)
    print('... saved')


def save_learning_curve(history):
    """ save_learning_curve """
    x = range(EPOCHS)
    pyplot.plot(x, history.history['loss'], label="loss")
    pyplot.title("loss")
    pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    lc_name = 'LearningCurve'
    pyplot.savefig(lc_name + '.png')
    pyplot.close()


def predict(input_dir, gpu_num=None):
    (file_names, inputs) = load_images(input_dir, INPUT_IMAGE_SHAPE, with_normalize=WITH_NORM)
    network = M2Det(INPUT_IMAGE_SHAPE, BATCH_SIZE, class_num=CLASS_NUM)
    priors = network.get_prior_boxes()
    bbox_util = BBoxUtility(CLASS_NUM, priors)

    if isinstance(gpu_num, int):
        model = network.get_parallel_model(gpu_num)
    else:
        model = network.get_model()
#    model.summary()
    print('loading weghts ...')
    model.load_weights(os.path.join(DIR_MODEL, FILE_MODEL))
    print('... loaded')

    #"""
    print('predicting ...')
    preds = model.predict(inputs, BATCH_SIZE)
    print('... predicted')

    print('result saveing ...')
    pred_pbox = preds[0, :, -8:]
    results = bbox_util.detection_out(preds)
    image_data = __outputs_to_image_data(inputs, results, file_names)
    save_images(DIR_OUTPUTS, image_data, file_names, with_unnormalize=WITH_NORM)
    print('... finish .')
    #"""


class Pred():
    #def __init__(self, score, label, xmin, xmax, ymin, ymax):
    def __init__(self, score, label, xmin, xmax, ymin, ymax
                 , org_xmin, org_ymin, org_xmax, org_ymax
                 , pb_xmin, pb_ymin, pb_xmax, pb_ymax
                 , var_1, var_2, var_3, var_4):
        self.score = score
        self.label = label
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.org_xmin = org_xmin
        self.org_ymin = org_ymin
        self.org_xmax = org_xmax
        self.org_ymax = org_ymax
        self.pb_xmin = pb_xmin
        self.pb_ymin = pb_ymin
        self.pb_xmax = pb_xmax
        self.pb_ymax = pb_ymax
        self.var_1 = var_1
        self.var_2 = var_2
        self.var_3 = var_3
        self.var_4 = var_4


#def __outputs_to_image_data(images, preds):
def __outputs_to_image_data(images, preds, filenames):
    # TODO : Refactoring
    image_data = []
    for i, img in enumerate(images):
        filename = filenames[i]
        if len(preds[i]) < 1:
            print('no pred : ', filename)
            image_data.append(img)
            continue
        # Parse the outputs.
        det_label = preds[i][:, 0]
        det_conf = preds[i][:, 1]
        det_xmin = preds[i][:, 2]
        det_ymin = preds[i][:, 3]
        det_xmax = preds[i][:, 4]
        det_ymax = preds[i][:, 5]

        org_xmin = preds[i][:, 6]
        org_ymin = preds[i][:, 7]
        org_xmax = preds[i][:, 8]
        org_ymax = preds[i][:, 9]

        org_pb_xmin = preds[i][:, 10]
        org_pb_ymin = preds[i][:, 11]
        org_pb_xmax = preds[i][:, 12]
        org_pb_ymax = preds[i][:, 13]

        org_var_1 = preds[i][:, 14]
        org_var_2 = preds[i][:, 15]
        org_var_3 = preds[i][:, 16]
        org_var_4 = preds[i][:, 17]

        np.set_printoptions(threshold=10000000)

        ## Get detections with confidence higher than 0.6.
        #top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
        #if len(top_indices) < 1:
        #    print('[%03d]' % i, ' top_confs is 0 (all confs is ', len(det_conf), ')')

        top_conf = det_conf #[top_indices]
        top_label_indices = det_label.tolist() #det_label[top_indices].tolist()
        top_xmin = det_xmin #[top_indices]
        top_ymin = det_ymin #[top_indices]
        top_xmax = det_xmax #[top_indices]
        top_ymax = det_ymax #[top_indices]

        #colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()
        col = [0, 126, 252]
        divider = top_conf.shape[0]
        if divider > 1:
            if divider > 100:
                divider = 100
            col = [i for i in range(255)[::(255 // divider - 1)]]

        pred_list = []
        for j in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[j] * img.shape[1]))
            ymin = int(round(top_ymin[j] * img.shape[0]))
            xmax = int(round(top_xmax[j] * img.shape[1]))
            ymax = int(round(top_ymax[j] * img.shape[0]))
            score = top_conf[j]
            label = int(top_label_indices[j])
            if label > 0:
                if xmin >= 0 and ymin >= 0:
                    pred_list.append(Pred(score, label, xmin, xmax, ymin, ymax
                                          , org_xmin[j], org_ymin[j], org_xmax[j], org_ymax[j]
                                          , org_pb_xmin[j], org_pb_ymin[j], org_pb_xmax[j], org_pb_ymax[j]
                                          , org_var_1[j], org_var_2[j], org_var_3[j], org_var_4[j]
                                         ))


        #print('&&&&&&&&&& : ', filename)

        pred_list_sort = pred_list.copy()
        pred_list_sort.sort(key=lambda p: p.score)
        pred_list_sort.reverse()
        for k, p in enumerate(pred_list_sort):
            if p.score < 0.6:
                break
            #print('@@@@@ %03d' % k)
            #print('### label : ', p.label)
            #print('### conf(score) : ', p.score)
            #print('### (xmin, ymin, xmax, ymax) : ', (p.xmin, p.ymin, p.xmax, p.ymax))
            #print('### (org_xmin, org_ymin, org_xmax, org_ymax) : ', (p.org_xmin, p.org_ymin, p.org_xmax, p.org_ymax))
            #print('### (pb_xmin, pb_ymin, pb_xmax, pb_ymax) : ', (p.pb_xmin, p.pb_ymin, p.pb_xmax, p.pb_ymax))
            #print('### (var_1, var_2, var_3, var_4) : ', (p.var_1, p.var_2, p.var_3, p.var_4))
            #print('')

            caption = '%d : %1.2f' % (p.label, p.score)
            reg_lt = (p.xmin, p.ymin) # (p.ymin, p.xmin)
            reg_rb = (p.xmax, p.ymax) # (p.ymax, p.xmax)
            #print('%02d - %02d : ' % (i, k), '[%02d] %f' % (p.label, p.score), '  (reg_lt, reg_rb)=', (reg_lt, reg_rb))
            c_k = k
            if c_k >= len(col):
                c_k = c_k - (len(col)) * (c_k // (len(col)))
            color = (col[c_k], col[::-1][c_k], 0) # colors[label]
            cv2.putText(img, caption, reg_lt, cv2.FONT_HERSHEY_PLAIN, 1, color)
            cv2.rectangle(img, reg_lt, reg_rb, color, 2)
        image_data.append(img)
        #print('$$$$$$$$$$$$$$$$$$$$')
        #print('')
    return image_data


if __name__ == '__main__':
    args = get_option(EPOCHS)
    EPOCHS = args.epoch

    if not(os.path.exists(DIR_MODEL)):
        os.mkdir(DIR_MODEL)
    if not(os.path.exists(DIR_OUTPUTS)):
        os.mkdir(DIR_OUTPUTS)

    train(gpu_num=GPU_NUM, with_generator=False, load_model=False)
    #train(gpu_num=GPU_NUM, with_generator=True, load_model=False)

    #predict(DIR_INPUTS, gpu_num=GPU_NUM)
    #predict(DIR_PREDICTS, gpu_num=GPU_NUM)
