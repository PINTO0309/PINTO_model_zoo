### tf-nightly-2.2.0.dev20200428

import tensorflow as tf
import numpy as np

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('deeplab_v3_plus_mnv3_decoder_256_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - deeplab_v3_plus_mnv3_decoder_256_weight_quant.tflite")