import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import os
import glob


# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_16_225')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('posenet_resnet50_16_225_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - posenet_resnet50_16_225_float16_quant.tflite")

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_16_257')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('posenet_resnet50_16_257_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - posenet_resnet50_16_257_float16_quant.tflite")

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_16_321')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('posenet_resnet50_16_321_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - posenet_resnet50_16_321_float16_quant.tflite")

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_16_385')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('posenet_resnet50_16_385_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - posenet_resnet50_16_385_float16_quant.tflite")

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_16_513')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('posenet_resnet50_16_513_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - posenet_resnet50_16_513_float16_quant.tflite")



# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_32_225')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('posenet_resnet50_32_225_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - posenet_resnet50_32_225_float16_quant.tflite")

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_32_257')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('posenet_resnet50_32_257_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - posenet_resnet50_32_257_float16_quant.tflite")

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_32_321')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('posenet_resnet50_32_321_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - posenet_resnet50_32_321_float16_quant.tflite")

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_32_385')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('posenet_resnet50_32_385_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - posenet_resnet50_32_385_float16_quant.tflite")

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_32_513')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('posenet_resnet50_32_513_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - posenet_resnet50_32_513_float16_quant.tflite")