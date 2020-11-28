### tf_nightly-2.5.0.dev20201123

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

height = 512
width  = 512

def representative_dataset_gen():
  for data in raw_test_data.take(10):
    image = data['image'].numpy()
    image = tf.image.resize(image, (height, width))
    image = image[np.newaxis,:,:,:]
    image = image - 127.5
    image = image * 0.007843
    yield [image]

raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.representative_dataset = representative_dataset_gen
tflite_model = converter.convert()
with open('efficientdet_d0_{}x{}_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Integer Quantization complete! - efficientdet_d0_{}x{}_integer_quant.tflite'.format(height, width))
