import os
import sys
from glob import glob
from PIL import Image
import numpy as np
import time

def resize(all_file):

  tmp = None
  height, width = 256, 256
  for i, f in enumerate(all_file):
    img = Image.open(f)
    img = img.resize(size=(height, width))
    nparray = np.array(img).astype(np.float32)
    nparray = nparray[np.newaxis, :, :, :]

    if tmp is not None:
        tmp = np.vstack((tmp, nparray))
        print("tmp.shape=", tmp.shape, f)
        np.save('calibration_data_img_coco_' + str(height) + 'x' + str(width), tmp)
    else:
        tmp = nparray.copy()
        print("tmp.shape=", tmp.shape, f)
        np.save('calibration_data_img_coco_' + str(height) + 'x' + str(width), tmp)

path = "./work/"
all_file = os.listdir(path)
all_file = [os.path.join(path, f) for f in all_file]
resize(all_file)
print("Finish!!")