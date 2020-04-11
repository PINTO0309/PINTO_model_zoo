### tf-nightly==2.2.0.dev20200410

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
import sys

#raw_test_data = np.load('calibration_data_img.npy')
#print("@@@@@@@@@@@@@@@@@@@ raw_test_data.shape=", raw_test_data.shape)
#for data in raw_test_data:
#    calibration_data = data[np.newaxis, :, :, :]
#    print("calibration_data.shape=", calibration_data.shape)
#sys.exit(0)

## Generating a calibration data set
def representative_dataset_gen():
    raw_test_data = np.load('calibration_data_img.npy')
    for data in raw_test_data:
        calibration_data = data[np.newaxis, :, :, :].astype(np.float32)
        calibration_data = (calibration_data - 0.45) / 0.225
        yield [calibration_data]


#tf.compat.v1.enable_eager_execution()

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#converter.experimental_new_converter = True
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_quant_model = converter.convert()
with open('./monodepth2_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - monodepth2_weight_quant.tflite")

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.experimental_new_converter = True
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('./monodepth2_integer_quant.tflite', 'wb') as w:
   w.write(tflite_quant_model)
print("Integer Quantization complete! - monodepth2_integer_quant.tflite")

# # Full Integer Quantization - Input/Output=int8
# converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# #converter.experimental_new_converter = True
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.representative_dataset = representative_dataset_gen
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# tflite_quant_model = converter.convert()
# with open('./monodepth2_full_integer_quant.tflite', 'wb') as w:
#    w.write(tflite_quant_model)
# print("Full Integer Quantization complete! - monodepth2_full_integer_quant.tflite")
