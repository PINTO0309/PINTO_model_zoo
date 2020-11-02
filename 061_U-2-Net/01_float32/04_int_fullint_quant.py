### tensorflow==2.3.1

import tensorflow as tf
import numpy as np
import sys
import tensorflow_datasets as tfds

def representative_dataset_gen():
    for data in raw_test_data.take(10):
        image = data['image'].numpy()
        image = tf.image.resize(image, (height, width))
        image = image[np.newaxis,:,:,:]
        image = image / 127.5 - 1.0
        yield [image]

raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)

height = 480
width  = 640

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('u2netp_{}x{}_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_quant_model)
print('Integer Quantization complete! - u2netp_{}x{}_integer_quant.tflite'.format(height, width))


# # Full Integer Quantization - Input/Output=int8
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('u2netp_{}x{}_full_integer_quant.tflite'.format(height, width), 'wb') as w:
#     w.write(tflite_quant_model)
# print('Integer Quantization complete! - u2netp_{}x{}_full_integer_quant.tflite'.format(height, width))

# # EdgeTPU
# import subprocess
# result = subprocess.check_output(["edgetpu_compiler", "-s", "u2netp_{}x{}_full_integer_quant.tflite".format(height, width)])
# print(result)
