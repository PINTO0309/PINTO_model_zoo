### tensorflow==2.3.1

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

height = 384
width  = 384

def representative_dataset_gen():
    for data in raw_test_data.take(10):
        image = data['image'].numpy()
        image = tf.image.resize(image, (height, width))
        image = image[np.newaxis,:,:,:]
        image = image / 127.5 - 1.0
        yield [image]

raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_model = converter.convert()
with open('midas_{}x{}_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Integer Quantization complete! - midas_{}x{}_integer_quant.tflite'.format(height, width))
