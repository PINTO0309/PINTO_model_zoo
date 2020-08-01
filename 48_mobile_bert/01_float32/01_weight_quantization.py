### tensorflow==2.3.0

import tensorflow as tf
import numpy as np

# Weight Quantization
converter = tf.lite.TFLiteConverter.from_saved_model('experiment/saved_model/1596242469')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('mobilebert_english_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - mobilebert_english_weight_quant.tflite")