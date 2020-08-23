### tf-nightly-2.2.0-dev20200502

import tensorflow as tf
import numpy as np

converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model_kitti')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('struct2depth_128x416_kitti_depth_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - struct2depth_128x416_kitti_depth_float16_quant.tflite")

converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model_cityscapes')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('struct2depth_128x416_cityscapes_depth_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - struct2depth_128x416_cityscapes_depth_float16_quant.tflite")