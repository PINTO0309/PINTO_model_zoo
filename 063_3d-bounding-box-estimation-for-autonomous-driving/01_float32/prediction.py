import os
import numpy as np
import cv2
import argparse
from utils.read_dir import ReadDir
from data_processing.KITTI_dataloader import KITTILoader
from utils.correspondece_constraint import *

import time

from config import config as cfg

if cfg().network == 'vgg16':
    from model import vgg16 as nn
if cfg().network == 'mobilenet_v2':
    from model import mobilenet_v2 as nn

def predict(args):
    # complie models
    model = nn.network()
    # model.load_weights('3dbox_weights_1st.hdf5')
    model.load_weights(args.w)

    # KITTI_train_gen = KITTILoader(subset='training')
    dims_avg, _ =KITTILoader(subset='training').get_average_dimension()

    # list all the validation images
    if args.a == 'training':
        all_imgs = sorted(os.listdir(test_image_dir))
        val_index = int(len(all_imgs)* cfg().split)
        val_imgs = all_imgs[val_index:]

    else:
        val_imgs = sorted(os.listdir(test_image_dir))

    start_time = time.time()

    for i in val_imgs:
        image_file = test_image_dir + i
        label_file = test_label_dir + i.replace('png', 'txt')
        prediction_file = prediction_path + i.replace('png', 'txt')
        calibration_file = test_calib_path + i.replace('png', 'txt')

        # write the prediction file
        with open(prediction_file, 'w') as predict:
            img = cv2.imread(image_file)
            img = np.array(img, dtype='float32')
            P2 = np.array([])
            for line in open(calibration_file):
                if 'P2' in line:
                    P2 = line.split(' ')
                    P2 = np.asarray([float(i) for i in P2[1:]])
                    P2 = np.reshape(P2, (3,4))

            for line in open(label_file):
                line = line.strip().split(' ')
                obj = detectionInfo(line)
                xmin = int(obj.xmin)
                xmax = int(obj.xmax)
                ymin = int(obj.ymin)
                ymax = int(obj.ymax)
                if obj.name in cfg().KITTI_cat:
                    # cropped 2d bounding box
                    if xmin == xmax or ymin == ymax:
                        continue
                    # 2D detection area
                    patch = img[ymin : ymax, xmin : xmax]
                    patch = cv2.resize(patch, (cfg().norm_h, cfg().norm_w))
                    patch -= np.array([[[103.939, 116.779, 123.68]]])
                    # extend it to match the training dimension
                    patch = np.expand_dims(patch, 0)

                    prediction = model.predict(patch)

                    dim = prediction[0][0]
                    bin_anchor = prediction[1][0]
                    bin_confidence = prediction[2][0]

                    # update with predict dimension
                    dims = dims_avg[obj.name] + dim
                    obj.h, obj.w, obj.l = np.array([round(dim, 2) for dim in dims])

                    # update with predicted alpha, [-pi, pi]
                    obj.alpha = recover_angle(bin_anchor, bin_confidence, cfg().bin)

                    # compute global and local orientation
                    obj.rot_global, rot_local = compute_orientaion(P2, obj)

                    # compute and update translation, (x, y, z)
                    obj.tx, obj.ty, obj.tz = translation_constraints(P2, obj, rot_local)

                    # output prediction label
                    output_line = obj.member_to_list()
                    output_line.append(1.0)
                    # Write regressed 3D dim and orientation to file
                    output_line = ' '.join([str(item) for item in output_line]) + '\n'
                    predict.write(output_line)
                    print('Write predicted labels for: ' + str(i))
    end_time = time.time()
    process_time = (end_time - start_time) / len(val_imgs)
    print(process_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for prediction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '-dir', type=str, default='/media/b920405/Windows/datasets/kitti_dateset/', help='File to predict')
    parser.add_argument('-a', '-dataset', type=str, default='tracklet', help='training dataset or tracklet')
    parser.add_argument('-w', '-weight', type=str, default='3dbox_mvnv2_320x320_1.4933_float32.h5', help ='Load trained weights')
    args = parser.parse_args()

    # Todo: subset = 'training' or 'tracklet'
    dir = ReadDir(args.d, subset=args.a,
                  tracklet_date='2011_09_26', tracklet_file='2011_09_26_drive_0084_sync')
    test_label_dir = dir.label_dir
    test_image_dir = dir.image_dir
    test_calib_path = dir.calib_dir
    prediction_path = dir.prediction_dir

    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)

    predict(args)