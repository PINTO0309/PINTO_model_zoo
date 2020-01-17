import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
import tensorflow_datasets as tfds

## Generating a calibration data set
def representative_dataset_gen():
  for data in raw_test_data.take(100):
    image = data['image'].numpy()
    image = tf.image.resize(image, (320, 320))
    image = image[np.newaxis,:,:,:]
    yield [image]

tf.compat.v1.enable_eager_execution()
#raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="./TFDS)
raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="./TFDS", download=False)

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./ssd_mobilenet_v3_large_coco_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - ssd_mobilenet_v3_large_coco_weight_quant.tflite")

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('./ssd_mobilenet_v3_large_coco_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - ssd_mobilenet_v3_large_coco_integer_quant.tflite")

# Full Integer Quantization - Input/Output=int8
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()
with open('./ssd_mobilenet_v3_large_coco_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Full Integer Quantization complete! - ssd_mobilenet_v3_large_coco_full_integer_quant.tflite")
