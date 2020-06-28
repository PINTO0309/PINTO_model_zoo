### tf-nightly-2.3.0.dev20200613

import tensorflow as tf

# # Float16 Quantization - Input/Output=float32
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_nhwc')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
# converter.target_spec.supported_types = [tf.float16]
# tflite_quant_model = converter.convert()
# with open('dbface_nhwc_256x256_float16_quant.tflite', 'wb') as w:
#     w.write(tflite_quant_model)
# print("Float16 Quantization complete! - dbface_nhwc_256x256_float16_quant.tflite")

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph('dbface_nhwc_striped_480x640.pb', ['x'], ['Identity','Identity_1','Identity_2'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open('dbface_480x640_float16_quant.tflite', "wb") as file:
    file.write(tflite_model)
print("Weight Quantization complete! - dbface_480x640_float16_quant.tflite")