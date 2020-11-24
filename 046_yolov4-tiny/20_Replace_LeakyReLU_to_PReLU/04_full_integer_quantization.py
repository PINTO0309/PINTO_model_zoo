### tf-nightly-2.5.0.dev20201124

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

height = 256
width  = 256

def representative_dataset_gen():
    for data in raw_test_data.take(100):
        image = data['image'].numpy()
        image = tf.image.resize(image, (height, width))
        image = image - 127.5
        image = image * 0.007843
        image = image[np.newaxis, :, :, :]
        yield [image]

raw_test_data, info = tfds.load(name="voc/2007", with_info=True, split="validation", data_dir="~/TFDS", download=False)

# Full Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_{}x{}'.format(height, width))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('yolov4_tiny_voc_{}x{}_full_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - yolov4_tiny_voc_{}x{}_full_integer_quant.tflite".format(height, width))
