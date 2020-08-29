### tf-nightly==2.4.0-dev20200829

import tensorflow as tf

# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_512x512')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('efficientdet_d0_512x512_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - efficientdet_d0_512x512_float16_quant.tflite")

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_416x416')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('efficientdet_d0_416x416_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - efficientdet_d0_416x416_float16_quant.tflite")

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_320x320')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('efficientdet_d0_320x320_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - efficientdet_d0_320x320_float16_quant.tflite")

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_256x256')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('efficientdet_d0_256x256_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - efficientdet_d0_256x256_float16_quant.tflite")