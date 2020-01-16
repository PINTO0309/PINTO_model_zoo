import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob

## Generating a calibration data set
def representative_dataset_gen():
    raw_test_data = np.load('calibration_data_desktop_sounds.npy')
    print("raw_test_data.shape=", raw_test_data.shape)
    for data in raw_test_data:
        calibration_data = data[np.newaxis, :, :, :]
        #print("calibration_data.shape=", calibration_data.shape)
        yield [calibration_data]

tf.compat.v1.enable_eager_execution()

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./mobilenetv2_fsd2018_41cls_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - mobilenetv2_fsd2018_41cls_weight_quant.tflite")

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('./mobilenetv2_fsd2018_41cls_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - mobilenetv2_fsd2018_41cls_integer_quant.tflite")

# Full Integer Quantization - Input/Output=int8
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()
with open('./mobilenetv2_fsd2018_41cls_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Full Integer Quantization complete! - mobilenetv2_fsd2018_41cls_full_integer_quant.tflite")
