### tensorflow==2.3.0

import tensorflow as tf
import numpy as np

# Float16 Quantization
converter = tf.lite.TFLiteConverter.from_saved_model('experiment/saved_model/1596242469')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('mobilebert_english_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - mobilebert_english_float16_quant.tflite")

