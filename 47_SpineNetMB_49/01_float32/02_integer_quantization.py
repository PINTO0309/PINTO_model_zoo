### tensorflow==2.3.0-rc2

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


def representative_dataset_gen():
  for data in raw_test_data.take(100):
    image = data['image'].numpy()
    image = tf.image.resize(image, (384, 384))
    image = image[np.newaxis,:,:,:]
    image = image - 127.5
    image = image * 0.007843
    yield [image]

raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('spinenetmb_49_384x384_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - spinenetmb_49_384x384_integer_quant.tflite")
