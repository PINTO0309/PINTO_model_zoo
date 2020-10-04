### tensorflow==2.3.1

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def representative_dataset_gen_256x256():
    for data in raw_test_data.take(10):
        image = data['image'].numpy()
        image = tf.image.resize(image, (256, 256))
        image = image[np.newaxis,:,:,:]
        image = image - 127.5
        image = image * 0.007843
        yield [image]

def representative_dataset_gen_320x320():
    for data in raw_test_data.take(100):
        image = data['image'].numpy()
        image = tf.image.resize(image, (320, 320))
        image = image[np.newaxis,:,:,:]
        image = image - 127.5
        image = image * 0.007843
        yield [image]



raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)

# Integer Quantization - Input/Output=float32
height = 256
width  = 256
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_256x256
tflite_model = converter.convert()
with open('3dbox_mbnv2_{}x{}_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Integer Quantization complete! - 3dbox_mbnv2_{}x{}_integer_quant.tflite'.format(height, width))

height = 320
width  = 320
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_320x320
tflite_model = converter.convert()
with open('3dbox_mbnv2_{}x{}_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Integer Quantization complete! - 3dbox_mbnv2_{}x{}_integer_quant.tflite'.format(height, width))
