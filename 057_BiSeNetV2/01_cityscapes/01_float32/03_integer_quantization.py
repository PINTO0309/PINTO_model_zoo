### tensorflow==2.3.0

import tensorflow as tf
import numpy as np

def representative_dataset_gen_256x256():
    for image in raw_test_data:
        image = tf.image.resize(image, (256, 256))
        image = image[np.newaxis,:,:,:]
        image = image - 127.5
        image = image * 0.007843
        yield [image]

def representative_dataset_gen_480x640():
    for image in raw_test_data:
        image = tf.image.resize(image, (480, 640))
        image = image[np.newaxis,:,:,:]
        image = image - 127.5
        image = image * 0.007843
        yield [image]

def representative_dataset_gen_512x1024():
    for image in raw_test_data:
        image = tf.image.resize(image, (512, 1024))
        image = image[np.newaxis,:,:,:]
        image = image - 127.5
        image = image * 0.007843
        yield [image]

raw_test_data = np.load('calibration_data_img_cityscapes.npy', allow_pickle=True)

# Integer Quantization - Input/Output=float32
height = 256
width  = 256
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_256x256
tflite_model = converter.convert()
with open('bisenetv2_cityscapes_{}x{}_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Weight Quantization complete! - bisenetv2_cityscapes_{}x{}_integer_quant.tflite'.format(height, width))

height = 480
width  = 640
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_480x640
tflite_model = converter.convert()
with open('bisenetv2_cityscapes_{}x{}_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Weight Quantization complete! - bisenetv2_cityscapes_{}x{}_integer_quant.tflite'.format(height, width))

height = 512
width  = 1024
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_512x1024
tflite_model = converter.convert()
with open('bisenetv2_cityscapes_{}x{}_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Weight Quantization complete! - bisenetv2_cityscapes_{}x{}_integer_quant.tflite'.format(height, width))