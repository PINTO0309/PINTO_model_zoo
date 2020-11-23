### tf_nightly-2.5.0.dev20201123

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

mean = np.asarray([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3) / 255
std  = np.asarray([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3) / 255

def representative_dataset_gen_320():
    for data in raw_test_data.take(100):
        image = data['image'].numpy()
        image = tf.image.resize(image, (320, 320))
        image = (image / 255 - mean) / std
        image = image[np.newaxis,:,:,:]
        yield [image]

def representative_dataset_gen_416():
    for data in raw_test_data.take(100):
        image = data['image'].numpy()
        image = tf.image.resize(image, (416, 416))
        image = (image / 255 - mean) / std
        image = image[np.newaxis,:,:,:]
        yield [image]

raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)

# Integer Quantization - Input/Output=float32
height = 320
width  = 320
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.representative_dataset = representative_dataset_gen_320
tflite_model = converter.convert()
with open('nanodet_{}x{}_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Integer Quantization complete! - nanodet_{}x{}_integer_quant.tflite'.format(height, width))


height = 416
width  = 416
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.representative_dataset = representative_dataset_gen_416
tflite_model = converter.convert()
with open('nanodet_{}x{}_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Integer Quantization complete! - nanodet_{}x{}_integer_quant.tflite'.format(height, width))