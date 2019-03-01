import os
import glob
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def load_images(dir_name, image_shape, with_normalize=True):
    files = glob.glob(os.path.join(dir_name, '*.png'))
    files.sort()
    h, w, ch = image_shape
    images = []

    pbar = tqdm(total=len(files), desc="load_images", unit=" Files")
    for i, file in enumerate(files):
        img = load_image(file, image_shape, with_normalize=with_normalize)
        images.append(img)
        pbar.update(1)
    pbar.close()
    return (files, np.array(images, dtype=np.float32))


def load_image(file_name, image_shape, with_normalize=True, is_binary=False):
    #src_img = Image.open(file)
    #dist_img = src_img.convert('L')
    #img_array = np.asarray(dist_img)
    #img_array = np.reshape(img_array, image_shape)
    #images.append(img_array / 255)
    src_img = cv2.imread(file_name)
    if src_img is None:
        return None

    dist_img = src_img
    if not is_binary and image_shape[2] == 1:
        dist_img = cv2.cvtColor(dist_img, cv2.COLOR_BGR2GRAY)

    if is_binary:
        h, w, ch = np.shape(dist_img)
        dist_img_tmp = np.zeros((h, w, 1))
        for h_ in range(h):
            for w_ in range(w):
                if dist_img[h_][w_][0] == 0 and dist_img[h_][w_][1] == 0 and dist_img[h_][w_][2] == 0:
                    continue
                dist_img_tmp[h_][w_] = 255
        dist_img = dist_img_tmp

    dist_img = cv2.resize(dist_img, (image_shape[0], image_shape[1]))
    if with_normalize:
        dist_img = dist_img / 255
    return dist_img


def save_images(dir_name, image_data_list, file_name_list, with_unnormalize=True):
    for _, (image_data, file_name) in enumerate(zip(image_data_list, file_name_list)):
        name = os.path.basename(file_name)
        #(w, h, _) = image_data.shape
        #image_data = np.reshape(image_data, (w, h))
        #distImg = Image.fromarray(image_data * 255)
        #distImg = distImg.convert('RGB')
        name_base, ext = os.path.splitext(name)
        save_path = os.path.join(dir_name, name_base + ext)
        save_image(image_data, save_path, with_unnormalize=with_unnormalize)

def save_image(image_data, save_path, with_unnormalize=True, binary_threshold=None):
    if with_unnormalize:
        image_data *= 255
    if isinstance(binary_threshold, int):
        image_data[image_data < binary_threshold] = 0
        image_data[image_data != 0] = 255
    img = image_data.astype(np.uint8)
    cv2.imwrite(save_path, img)
    print('saved : ' , save_path)
