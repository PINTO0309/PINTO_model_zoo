### tensorflow==2.3.0

import tensorflow as tf

# Weight Quantization - Input/Output=float32
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph('dbface_nhwc_480x640.pb', ['x'], ['Identity','Identity_1','Identity_2'])
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
tflite_quant_model = converter.convert()
with open('dbface_480x640_float32.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("TFLite convert complete! - dbface_480x640_float32.tflite")
