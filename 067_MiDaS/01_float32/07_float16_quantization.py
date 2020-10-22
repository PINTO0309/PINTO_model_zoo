### tensorflow==2.3.1

import tensorflow as tf

# Float16 Quantization - Input/Output=float32
height = 384
width  = 384
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open('midas_{}x{}_float16_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Float16 Quantization complete! - midas_{}x{}_float16_quant.tflite'.format(height, width))
