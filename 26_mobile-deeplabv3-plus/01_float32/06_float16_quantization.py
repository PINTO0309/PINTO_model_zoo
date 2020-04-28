### tf-nightly-2.2.0.dev20200428

import tensorflow as tf
import os
import glob

# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('deeplab_v3_plus_mnv3_decoder_256_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - deeplab_v3_plus_mnv3_decoder_256_float16_quant.tflite")

