# Tensorflow v2.1.0

import tensorflow as tf
import numpy as np
import skimage
from skimage.transform import resize
import skimage.color as sc
import imageio
import os
import re

def representative_dataset_gen():
  for index, calculate_lr_img in enumerate(calculate_lr_imgs):
    print(index, "calculate_lr_img.shape=", calculate_lr_img.shape)
    ypbpr = sc.rgb2ypbpr(calculate_lr_img / 255.0) # RGB to YPbPr
    x_scale = resize(calculate_lr_img, [320, 480]) # Resize (Height, Width)

    y = ypbpr[..., 0].astype(np.float32)
    y = np.expand_dims(y, axis=-1) # Add a dimension to the end
    y = y[np.newaxis, :, :, :]
    pbpr = sc.rgb2ypbpr(x_scale / 255)[..., 1:].astype(np.float32) # Extract backward from INDEX = 1 of last dimension
    pbpr = pbpr[np.newaxis, :, :, :]

    #print("y.shape=", y.shape)
    #print("pbpr.shape=", pbpr.shape)

    #yield [pbpr, y]
    yield [y, pbpr]

def load_file_list(path=None, regx='\\.jpg', printable=True, keep_prefix=False):
    if path is None:
        path = os.getcwd()
    file_list = os.listdir(path)
    return_list = []
    for _, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(f)
    if keep_prefix:
        for i, f in enumerate(return_list):
            return_list[i] = os.path.join(path, f)
    return return_list


tf.compat.v1.enable_eager_execution()

calculate_lr_img_list = sorted(load_file_list(path='./B100', regx='.*LR\\.\\w+g', printable=False, keep_prefix=True))
calculate_lr_imgs = [imageio.imread(p, pilmode='RGB') for p in calculate_lr_img_list]

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()
with open('./FALSR_A_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - FALSR_A_full_integer_quant.tflite")
