### tensorflow==2.3.0

import tensorflow as tf

# Float16 Quantization - Input/Output=float32
height = 256
width  = 256
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open('bisenetv2_celebamaskhq_{}x{}_float16_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Weight Quantization complete! - bisenetv2_celebamaskhq_{}x{}_float16_quant.tflite'.format(height, width))

height = 448
width  = 448
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open('bisenetv2_celebamaskhq_{}x{}_float16_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Weight Quantization complete! - bisenetv2_celebamaskhq_{}x{}_float16_quant.tflite'.format(height, width))

height = 480
width  = 640
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open('bisenetv2_celebamaskhq_{}x{}_float16_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Weight Quantization complete! - bisenetv2_celebamaskhq_{}x{}_float16_quant.tflite'.format(height, width))