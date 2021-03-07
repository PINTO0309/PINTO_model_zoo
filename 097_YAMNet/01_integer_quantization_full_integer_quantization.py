### tf_nightly==2.5.0-dev20210306

import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def representative_dataset_gen():
    yield [raw_test_data.astype(np.float32)]

raw_test_data = np.load('miaow_16k.npy')

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('.')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.representative_dataset = representative_dataset_gen
tflite_model = converter.convert()
with open('model_integer_quant.tflite', 'wb') as w:
    w.write(tflite_model)
print('Integer Quantization complete! - model_integer_quant.tflite')

# # Full Integer Quantization - Input/Output=float32
# converter = tf.lite.TFLiteConverter.from_saved_model('.')
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.representative_dataset = representative_dataset_gen
# tflite_model = converter.convert()
# with open('model_full_integer_quant.tflite', 'wb') as w:
#     w.write(tflite_model)
# print('Integer Quantization complete! - model_full_integer_quant.tflite')