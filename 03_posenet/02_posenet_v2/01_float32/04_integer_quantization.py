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

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_16_225')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('posenet_resnet50_16_225_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - posenet_resnet50_16_225_integer_quant.tflite")

# # Integer Quantization - Input/Output=float32
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_16_257')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('posenet_resnet50_16_257_integer_quant.tflite', 'wb') as w:
#     w.write(tflite_quant_model)
# print("Integer Quantization complete! - posenet_resnet50_16_257_integer_quant.tflite")

# # Integer Quantization - Input/Output=float32
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_16_321')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('posenet_resnet50_16_321_integer_quant.tflite', 'wb') as w:
#     w.write(tflite_quant_model)
# print("Integer Quantization complete! - posenet_resnet50_16_321_integer_quant.tflite")

# # Integer Quantization - Input/Output=float32
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_16_385')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('posenet_resnet50_16_385_integer_quant.tflite', 'wb') as w:
#     w.write(tflite_quant_model)
# print("Integer Quantization complete! - posenet_resnet50_16_385_integer_quant.tflite")

# # Integer Quantization - Input/Output=float32
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_16_513')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('posenet_resnet50_16_513_integer_quant.tflite', 'wb') as w:
#     w.write(tflite_quant_model)
# print("Integer Quantization complete! - posenet_resnet50_16_513_integer_quant.tflite")



# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_32_225')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('posenet_resnet50_32_225_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - posenet_resnet50_32_225_integer_quant.tflite")

# # Integer Quantization - Input/Output=float32
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_32_257')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('posenet_resnet50_32_257_integer_quant.tflite', 'wb') as w:
#     w.write(tflite_quant_model)
# print("Integer Quantization complete! - posenet_resnet50_32_257_integer_quant.tflite")

# # Integer Quantization - Input/Output=float32
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_32_321')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('posenet_resnet50_32_321_integer_quant.tflite', 'wb') as w:
#     w.write(tflite_quant_model)
# print("Integer Quantization complete! - posenet_resnet50_32_321_integer_quant.tflite")

# # Integer Quantization - Input/Output=float32
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_32_385')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('posenet_resnet50_32_385_integer_quant.tflite', 'wb') as w:
#     w.write(tflite_quant_model)
# print("Integer Quantization complete! - posenet_resnet50_32_385_integer_quant.tflite")

# # Integer Quantization - Input/Output=float32
# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_posenet_resnet50_32_513')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('posenet_resnet50_32_513_integer_quant.tflite', 'wb') as w:
#     w.write(tflite_quant_model)
# print("Integer Quantization complete! - posenet_resnet50_32_513_integer_quant.tflite")