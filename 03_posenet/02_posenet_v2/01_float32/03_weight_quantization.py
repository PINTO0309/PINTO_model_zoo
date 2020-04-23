import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

#def representative_dataset_gen():
#  for data in raw_test_data.take(100):
#    image = data['image'].numpy()
#    image = tf.image.resize(image, (320, 320))
#    image = image[np.newaxis,:,:,:]
#    yield [image]

tf.compat.v1.enable_eager_execution()

## Generating a calibration data set
##raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="./TFDS")
#raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="./TFDS", download=False)
#print(info)

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./0')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./model-mobilenet_v1_101_225_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - model-mobilenet_v1_101_225_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./0')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./model-mobilenet_v1_101_257_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - model-mobilenet_v1_101_257_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./0')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./model-mobilenet_v1_101_321_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - model-mobilenet_v1_101_321_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./0')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./model-mobilenet_v1_101_385_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - model-mobilenet_v1_101_385_weight_quant.tflite")

# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./0')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./model-mobilenet_v1_101_513_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - model-mobilenet_v1_101_513_weight_quant.tflite")
