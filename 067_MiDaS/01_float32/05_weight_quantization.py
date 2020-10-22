### tensorflow==2.3.1

import tensorflow as tf

# Weight Quantization - Input/Output=float32
height = 384
width  = 384
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
with open('midas_{}x{}_weight_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Weight Quantization complete! - midas_{}x{}_weight_quant.tflite'.format(height, width))
