### tensorflow==2.2.0

import tensorflow as tf

# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_mobilenet_v1_100_257x257_multi_kpt_stripped')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('posenet_mobilenet_v1_100_257x257_multi_kpt_stripped_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - posenet_mobilenet_v1_100_257x257_multi_kpt_stripped_float16_quant.tflite")
