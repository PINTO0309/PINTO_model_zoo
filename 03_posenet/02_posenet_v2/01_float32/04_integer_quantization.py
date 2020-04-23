import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import os
import glob

## Generating a calibration data set
def representative_dataset_gen():
    folder = ["images"]
    image_size = 225
    raw_test_data = []
    for name in folder:
        dir = "./" + name
        files = glob.glob(dir + "/*.jpg")
        for file in files:
            image = Image.open(file)
            image = image.convert("RGB")
            image = image.resize((image_size, image_size))
            image = np.asarray(image).astype(np.float32)
            image = image[np.newaxis,:,:,:]
            raw_test_data.append(image)

    for data in raw_test_data:
        yield [data]


tf.compat.v1.enable_eager_execution()

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./0')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('./model-mobilenet_v1_101_225_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - model-mobilenet_v1_101_225_integer_quant.tflite")

## Integer Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./0')
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = representative_dataset_gen
#tflite_quant_model = converter.convert()
#with open('./model-mobilenet_v1_101_257_integer_quant.tflite', 'wb') as w:
#    w.write(tflite_quant_model)
#print("Integer Quantization complete! - model-mobilenet_v1_101_257_integer_quant.tflite")

## Integer Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./0')
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = representative_dataset_gen
#tflite_quant_model = converter.convert()
#with open('./model-mobilenet_v1_101_321_integer_quant.tflite', 'wb') as w:
#    w.write(tflite_quant_model)
#print("Integer Quantization complete! - model-mobilenet_v1_101_321_integer_quant.tflite")

## Integer Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./0')
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = representative_dataset_gen
#tflite_quant_model = converter.convert()
#with open('./model-mobilenet_v1_101_385_integer_quant.tflite', 'wb') as w:
#    w.write(tflite_quant_model)
#print("Integer Quantization complete! - model-mobilenet_v1_101_385_integer_quant.tflite")

## Integer Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./0')
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = representative_dataset_gen
#tflite_quant_model = converter.convert()
#with open('./model-mobilenet_v1_101_513_integer_quant.tflite', 'wb') as w:
#    w.write(tflite_quant_model)
#print("Integer Quantization complete! - model-mobilenet_v1_101_513_integer_quant.tflite")
