import os
import sys
from glob import glob
from PIL import Image
import numpy as np

def resize(all_file):

  tmp = None


  for idx, f in enumerate(all_file):
    img = Image.open(f)
    img_resize = img.resize((416, 128))
    #ftitle, fext = os.path.splitext(f)
    #img_resize.save(ftitle + '_half' + fext)

    nparray = np.array(img_resize)
    nparray = nparray[np.newaxis, :, :, :]

    if tmp is not None:
        tmp = np.vstack((tmp, nparray))
        print("tmp.shape=", tmp.shape)
        np.save('calibration_data_img_kitti', tmp)
    else:
        tmp = nparray.copy()
        print("tmp.shape=", tmp.shape)
        np.save('calibration_data_img_kitti', tmp)
    if idx >= 9:
      break

path = "./2011_09_29/2011_09_29_drive_0071_sync/image_02/data/"
all_file = os.listdir(path)
all_file = [os.path.join(path, f) for f in all_file]
resize(all_file)
print("Finish!!")