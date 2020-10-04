import copy
import cv2
import numpy as np
from config import config as cfg

def prepare_input_and_output(train_inst, image_dir):
    '''
    prepare image patch for training
    input:  train_inst -- input image for training
    output: img -- cropped bbox
            train_inst['dims'] -- object dimensions
            train_inst['orient'] -- object orientation (or flipped orientation)
            train_inst['conf_flipped'] -- orientation confidence
    '''
    xmin = train_inst['xmin']  + np.random.randint(-cfg().jit, cfg().jit+1)
    ymin = train_inst['ymin']  + np.random.randint(-cfg().jit, cfg().jit+1)
    xmax = train_inst['xmax']  + np.random.randint(-cfg().jit, cfg().jit+1)
    ymax = train_inst['ymax']  + np.random.randint(-cfg().jit, cfg().jit+1)

    img = cv2.imread(image_dir)

    if cfg().jit != 0:
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, img.shape[1] - 1)
        ymax = min(ymax, img.shape[0] - 1)

        if xmin < 0 or xmin >= img.shape[1]:
            xmin = 0
        if ymin < 0 or ymin >= img.shape[0]:
            ymin = 0

        if xmax < 0:
            xmax = img.shape[1] - 1
        if ymax < 0:
            ymax = img.shape[0] - 1

        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin

    img = copy.deepcopy(img[ymin:ymax + 1, xmin:xmax + 1]).astype(np.float32)

    # flip the image
    # 50% percent choose 1, 50% percent choose 0
    flip = np.random.binomial(1, .5)
    if flip > 0.5:
        img = cv2.flip(img, 1)

    # resize the image to standard size
    # print('@@@@@@@@@@@@@@@@@@@@ image_dir {} ymin:ymax {}:{} xmin:xmax {}:{}'.format(image_dir, ymin, ymax, xmin, xmax))
    # print('@@@@@@@@@@@@@@@@@@@@ img.shape, cfg().norm_h, cfg().norm_w', img.shape, cfg().norm_h, cfg().norm_w)
    img = cv2.resize(img, (cfg().norm_h, cfg().norm_w))
    # minus the mean value in each channel
    img = img - np.array([[[103.939, 116.779, 123.68]]])

    ### Fix orientation and confidence
    if flip > 0.5:
        return img, train_inst['dims'], train_inst['orient_flipped'], train_inst['conf_flipped']
    else:
        return img, train_inst['dims'], train_inst['orient'], train_inst['conf']


def data_gen(all_objs):
    '''
    generate data for training
    input: all_objs -- all objects used for training
           batch_size -- number of images used for training at once
    yield: x_batch -- (batch_size, 224, 224, 3),  input images to training process at each batch
           d_batch -- (batch_size, 3),  object dimensions
           o_batch -- (batch_size, 2, 2), object orientation
           c_batch -- (batch_size, 2), angle confidence
    '''
    num_obj = len(all_objs)

    keys = list(range(num_obj))
    np.random.shuffle(keys)

    l_bound = 0
    r_bound = cfg().batch_size if cfg().batch_size < num_obj else num_obj

    while True:
        if l_bound == r_bound:
            l_bound = 0
            r_bound = cfg().batch_size if cfg().batch_size < num_obj else num_obj
            np.random.shuffle(keys)

        currt_inst = 0
        x_batch = np.zeros((r_bound - l_bound, cfg().norm_h, cfg().norm_w, 3))
        d_batch = np.zeros((r_bound - l_bound, 3))
        o_batch = np.zeros((r_bound - l_bound, cfg().bin, 2))
        c_batch = np.zeros((r_bound - l_bound, cfg().bin))

        for key in keys[l_bound:r_bound]:
            # augment input image and fix object's orientation and confidence
            image, dimension, orientation, confidence = prepare_input_and_output(all_objs[key], all_objs[key]['image'],
                                                                                 )

            x_batch[currt_inst, :] = image
            d_batch[currt_inst, :] = dimension
            o_batch[currt_inst, :] = orientation
            c_batch[currt_inst, :] = confidence

            currt_inst += 1

        yield x_batch, [d_batch, o_batch, c_batch]

        l_bound = r_bound
        r_bound = r_bound + cfg().batch_size

        if r_bound > num_obj:
            r_bound = num_obj
