### tensorflow==2.2.0

import tensorflow as tf

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenet_v1_100_257x257_multi_kpt_stripped')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('posenet_mobilenet_v1_100_257x257_multi_kpt_stripped_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - posenet_mobilenet_v1_100_257x257_multi_kpt_stripped_weight_quant.tflite")
