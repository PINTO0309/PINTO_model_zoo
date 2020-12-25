import tensorflow as tf

# No Quantization - Input/Output=float32
print('tflite Float32 convertion started', '=' * 51)
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
with open('saved_model/model_float32.tflite', 'wb') as w:
    w.write(tflite_model)
print('tflite Float32 convertion complete! - saved_model/model_float32.tflite')

# Weight Quantization - Input/Output=float32
print('Weight Quantization started', '=' * 57)
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
with open('saved_model/model_weight_quant.tflite', 'wb') as w:
    w.write(tflite_model)
print('Weight Quantization convertion complete! - saved_model/model_weight_quant.tflite')

# Float16 Quantization - Input/Output=float32
print('Float16 Quantization started', '=' * 56)
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_quant_model = converter.convert()
with open('saved_model/model_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print('Float16 Quantization complete! - saved_model/model_float16_quant.tflite')
