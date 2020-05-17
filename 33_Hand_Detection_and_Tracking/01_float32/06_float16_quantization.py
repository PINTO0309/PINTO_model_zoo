### tensorflow==2.2.0

import tensorflow as tf

# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_hand_landmark')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('hand_landmark_256_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - hand_landmark_256_float16_quant.tflite")

# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_hand_landmark_3d')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('hand_landmark_3d_256_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - hand_landmark_3d_256_float16_quant.tflite")

# # Float16 Quantization - Input/Output=float32
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_palm_detection_without_custom_op')
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# tflite_quant_model = converter.convert()
# with open('palm_detection_without_custom_op_256_float16_quant.tflite', 'wb') as w:
#     w.write(tflite_quant_model)
# print("Weight Quantization complete! - palm_detection_without_custom_op_256_float16_quant.tflite")