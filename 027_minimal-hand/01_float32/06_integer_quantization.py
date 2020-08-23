### tf-nightly-2.2.0-dev20200429

import tensorflow as tf
import numpy as np

def representative_dataset_gen_hand():
  for image in raw_test_data_hand:
    image = tf.image.resize(image.astype(np.float32), (128, 128))
    image = image[np.newaxis,:,:,:]
    image = image / 255
    yield [image]

def representative_dataset_gen_joint():
  for data in raw_test_data_joint:
    data = data[np.newaxis,:,:].astype(np.float32)
    yield [data]

raw_test_data_hand = np.load('hand_dataset.npy', allow_pickle=True)
raw_test_data_joint = np.load('joint_dataset.npy', allow_pickle=True)

# tf.compat.v1.enable_eager_execution()

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_detnet')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
# converter.experimental_new_converter = True
converter.representative_dataset = representative_dataset_gen_hand
tflite_quant_model = converter.convert()
with open('detnet_128_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - detnet_128_integer_quant.tflite")

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_iknet')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
# converter.experimental_new_converter = True
converter.representative_dataset = representative_dataset_gen_joint
tflite_quant_model = converter.convert()
with open('iknet_84_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - iknet_84_integer_quant.tflite")