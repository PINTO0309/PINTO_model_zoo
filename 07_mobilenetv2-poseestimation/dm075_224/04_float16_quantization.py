### tensorflow==2.2.0

import tensorflow as tf

# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('mobilenet_v2_pose_368_432_dm075_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - mobilenet_v2_pose_368_432_dm075_float16_quant.tflite")