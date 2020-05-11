### tensorflow==2.2.0

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import os
import glob

def representative_dataset_gen():
  for data in raw_test_data.take(100):
    image = data['image'].numpy()
    image = tf.image.resize(image, (128, 128))
    image = image[np.newaxis,:,:,:]
    yield [image]

raw_test_data, info = tfds.load(name="the300w_lp", with_info=True, split="train", data_dir="~/TFDS", download=False)

# Full Integer Quantization - Input/Output=uint8
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()
with open('face_detection_front_128_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Full Integer Quantization complete! - face_detection_front_128_full_integer_quant.tflite")
