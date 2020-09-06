### tf-nightly==2.4.0-dev20200905

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

# Full Integer Quantization - Input/Output=float32
height = 256
width  = 256
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}_fullint'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen_256x256
tflite_model = converter.convert()
with open('bisenetv2_cityscapes_{}x{}_full_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Full Integer Quantization complete! - bisenetv2_cityscapes_{}x{}_full_integer_quant.tflite'.format(height, width))

height = 480
width  = 640
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}_fullint'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen_480x640
tflite_model = converter.convert()
with open('bisenetv2_cityscapes_{}x{}_full_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Full Integer Quantization complete! - bisenetv2_cityscapes_{}x{}_full_integer_quant.tflite'.format(height, width))

height = 512
width  = 1024
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}_fullint'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen_512x1024
tflite_model = converter.convert()
with open('bisenetv2_cityscapes_{}x{}_full_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Full Integer Quantization complete! - bisenetv2_cityscapes_{}x{}_full_integer_quant.tflite'.format(height, width))

