### tensorflow==2.2.0

import tensorflow as tf

# # Weight Quantization - Input/Output=float32
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
# # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
# tflite_quant_model = converter.convert()
# with open('dbface_256x256_weight_quant.tflite', 'wb') as w:
#     w.write(tflite_quant_model)
# print("Weight Quantization complete! - dbface_256x256_weight_quant.tflite")

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph('dbface_nhwc_striped_480x640.pb', ['x'], ['Identity','Identity_1','Identity_2'])
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
with open('dbface_480x640_weight_quant.tflite', "wb") as file:
    file.write(tflite_model)
print("Weight Quantization complete! - dbface_480x640_weight_quant.tflite")