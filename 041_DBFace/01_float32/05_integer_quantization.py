### tensorflow==2.2.0

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def representative_dataset_gen():
  for data in raw_test_data.take(100):
    image = data['image'].numpy()
    image = tf.image.resize(image, (480, 640))
    image = image[np.newaxis,:,:,:]
    yield [image]

raw_test_data, info = tfds.load(name="the300w_lp", with_info=True, split="train", data_dir="~/TFDS", download=False)

# # Integer Quantization - Input/Output=float32
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_tf')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('dbface_tf_512x512_integer_quant.tflite', 'wb') as w:
#     w.write(tflite_quant_model)
# print("Integer Quantization complete! - dbface_tf_512x512_integer_quant.tflite")


converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph('dbface_nhwc_striped_480x640.pb', ['x'], ['Identity','Identity_1','Identity_2'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
converter.representative_dataset = representative_dataset_gen
tflite_model = converter.convert()
with open('dbface_480x640_integer_quant.tflite', "wb") as file:
    file.write(tflite_model)
print("Weight Quantization complete! - dbface_480x640_integer_quant.tflite")