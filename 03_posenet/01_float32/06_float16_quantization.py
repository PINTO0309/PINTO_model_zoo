import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import os
import glob

tf.compat.v1.enable_eager_execution()

## Float16 Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./0')
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
#tflite_quant_model = converter.convert()
#with open('./model-mobilenet_v1_101_225_float16_quant.tflite', 'wb') as w:
#    w.write(tflite_quant_model)
#print("Integer Quantization complete! - model-mobilenet_v1_101_225_float16_quant.tflite")

## Float16 Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./0')
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
#tflite_quant_model = converter.convert()
#with open('./model-mobilenet_v1_101_257_float16_quant.tflite', 'wb') as w:
#    w.write(tflite_quant_model)
#print("Integer Quantization complete! - model-mobilenet_v1_101_257_float16_quant.tflite")

## Float16 Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./0')
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
#tflite_quant_model = converter.convert()
#with open('./model-mobilenet_v1_101_321_float16_quant.tflite', 'wb') as w:
#    w.write(tflite_quant_model)
#print("Integer Quantization complete! - model-mobilenet_v1_101_321_float16_quant.tflite")

## Float16 Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./0')
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
#tflite_quant_model = converter.convert()
#with open('./model-mobilenet_v1_101_385_float16_quant.tflite', 'wb') as w:
#    w.write(tflite_quant_model)
#print("Integer Quantization complete! - model-mobilenet_v1_101_385_float16_quant.tflite")

# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./0')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
tflite_quant_model = converter.convert()
with open('./model-mobilenet_v1_101_513_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - model-mobilenet_v1_101_513_float16_quant.tflite")
