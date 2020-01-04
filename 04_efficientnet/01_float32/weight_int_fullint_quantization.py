import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob

## Generating a calibration data set
def representative_dataset_gen():
    folder = ["images"]
    image_size = 224
    raw_test_data = []
    for name in folder:
        dir = "./" + name
        files = glob.glob(dir + "/*.JPEG")
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

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./efficientnet_b0_224_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - efficientnet_b0_224_weight_quant.tflite")

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('./efficientnet_b0_224_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - efficientnet_b0_224_integer_quant.tflite")

## Full Integer Quantization - Input/Output=int8
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = representative_dataset_gen
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.uint8
#converter.inference_output_type = tf.uint8
#tflite_quant_model = converter.convert()
#with open('./efficientnet_b0_224_full_integer_quant.tflite', 'wb') as w:
#    w.write(tflite_quant_model)
#print("Full Integer Quantization complete! - efficientnet_b0_224_full_integer_quant.tflite")
