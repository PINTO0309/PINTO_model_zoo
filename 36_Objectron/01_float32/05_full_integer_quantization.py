### tensorflow==2.3.0-rc1

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def representative_dataset_gen():
  for data in raw_test_data.take(100):
    image = data['image'].numpy()
    image = tf.image.resize(image, (640, 480))
    image = image[np.newaxis,:,:,:]
    image = image - 127.5
    image = image * 0.007843
    yield [image]

raw_test_data, info = tfds.load(name="voc/2007", with_info=True, split="validation", data_dir="~/TFDS", download=False)

# Full Integer Quantization - Input/Output=int8
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_object_detection_3d_chair')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('object_detection_3d_chair_640x480_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - object_detection_3d_chair_640x480_full_integer_quant.tflite")


# Full Integer Quantization - Input/Output=int8
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_object_detection_3d_sneakers')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('object_detection_3d_sneakers_640x480_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - object_detection_3d_sneakers_640x480_full_integer_quant.tflite")