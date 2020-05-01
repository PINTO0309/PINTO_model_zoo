### tf-nightly-2.2.0.dev20200428

import tensorflow as tf
import numpy as np
import os
import glob

def representative_dataset_gen():
    for image in raw_test_data:
        image = tf.image.resize(image, (256, 256))
        image = image[np.newaxis,:,:,:]
        image = image - 127.5
        image = image * 0.007843
        yield [image]


raw_test_data = np.load('person_dataset.npy', allow_pickle=True)

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('deeplab_v3_plus_mnv2_decoder_256_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - deeplab_v3_plus_mnv2_decoder_256_integer_quant.tflite")

