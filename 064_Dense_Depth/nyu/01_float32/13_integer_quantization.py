### tensorflow==2.3.1

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def representative_dataset_gen_480x640():
    for data in raw_test_data.take(10):
        image = data['image'].numpy()
        image = tf.image.resize(image, (480, 640))
        image = image[np.newaxis,:,:,:]
        image = image - 127.5
        image = image * 0.007843
        yield [image]


raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)

# Integer Quantization - Input/Output=float32
height = 480
width  = 640
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_nyu_{}x{}'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_480x640
tflite_model = converter.convert()
with open('dense_depth_nyu_{}x{}_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Integer Quantization complete! - dense_depth_nyu_{}x{}_integer_quant.tflite'.format(height, width))
