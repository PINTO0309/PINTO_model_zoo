### tf_nightly==2.5.0-dev20201204

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
std  = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

def representative_dataset_gen_256():
    for data in raw_test_data.take(10):
        image = data['image'].numpy()
        image = tf.image.resize(image, (256, 512))
        image = (image / 255 - mean) / std
        image = image[np.newaxis,:,:,:]
        yield [image]

def representative_dataset_gen_384():
    for data in raw_test_data.take(10):
        image = data['image'].numpy()
        image = tf.image.resize(image, (384, 768))
        image = (image / 255 - mean) / std
        image = image[np.newaxis,:,:,:]
        yield [image]

def representative_dataset_gen_512():
    for data in raw_test_data.take(10):
        image = data['image'].numpy()
        image = tf.image.resize(image, (512, 1024))
        image = (image / 255 - mean) / std
        image = image[np.newaxis,:,:,:]
        yield [image]


raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)

# Full Integer Quantization - Input/Output=float32
height = 256
width  = 512
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset_gen_256
tflite_model = converter.convert()
with open('erfnet_{}x{}_cityscapes_full_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Full Integer Quantization complete! - erfnet_{}x{}_cityscapes_full_integer_quant.tflite'.format(height, width))

height = 384
width  = 768
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset_gen_384
tflite_model = converter.convert()
with open('erfnet_{}x{}_cityscapes_full_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Full Integer Quantization complete! - erfnet_{}x{}_cityscapes_full_integer_quant.tflite'.format(height, width))


height = 512
width  = 1024
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset_gen_512
tflite_model = converter.convert()
with open('erfnet_{}x{}_cityscapes_full_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Full Integer Quantization complete! - erfnet_{}x{}_cityscapes_full_integer_quant.tflite'.format(height, width))