### tensorflow=2.1.0

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

raw_test_data = np.load('cityscapes_bielefeld_181.npy')

def representative_dataset_gen_257():
  data_count = 0
  for image in raw_test_data:
    #cp = Image.fromarray(np.uint8(image))
    #cp.save('test1.jpg')
    image = cv2.resize(image, (257, 257))
    #cv2.imwrite('test2.jpg', image)
    calibration_data = np.expand_dims(image, axis=0).astype(np.float32)
    data_count += 1
    print('257x257 Data index being processed = {0}'.format(data_count))
    yield [calibration_data]

def representative_dataset_gen_321():
  data_count = 0
  for image in raw_test_data:
    image = cv2.resize(image, (321, 321))
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    calibration_data = np.expand_dims(image, axis=0).astype(np.float32)
    data_count += 1
    print('321x321 Data index being processed = {0}'.format(data_count))
    yield [calibration_data]

def representative_dataset_gen_513():
  data_count = 0
  for image in raw_test_data:
    image = cv2.resize(image, (513, 513))
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    calibration_data = np.expand_dims(image, axis=0).astype(np.float32)
    data_count += 1
    print('513x513 Data index being processed = {0}'.format(data_count))
    yield [calibration_data]

def representative_dataset_gen_769():
  data_count = 0
  for image in raw_test_data:
    image = cv2.resize(image, (769, 769))
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    calibration_data = np.expand_dims(image, axis=0).astype(np.float32)
    data_count += 1
    print('769x769 Data index being processed = {0}'.format(data_count))
    yield [calibration_data]



#tf.compat.v1.enable_eager_execution()
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']

size = 257
graph_def_file="frozen_inference_graph_257_os16.pb"
input_tensor={"ImageTensor":[1,size,size,3]}
# Integer Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_257
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_slim_257_os16_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - edgetpu_deeplab_slim_257_os16_integer_quant.tflite")


size = 257
graph_def_file="frozen_inference_graph_257_os32.pb"
input_tensor={"ImageTensor":[1,size,size,3]}
# Integer Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_257
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_slim_257_os32_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - edgetpu_deeplab_slim_257_os32_integer_quant.tflite")


size = 321
graph_def_file="frozen_inference_graph_321_os16.pb"
input_tensor={"ImageTensor":[1,size,size,3]}
# Integer Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_321
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_slim_321_os16_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - edgetpu_deeplab_slim_321_os16_integer_quant.tflite")


size = 321
graph_def_file="frozen_inference_graph_321_os32.pb"
input_tensor={"ImageTensor":[1,size,size,3]}
# Integer Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_321
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_slim_321_os32_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - edgetpu_deeplab_slim_321_os32_integer_quant.tflite")


size = 513
graph_def_file="frozen_inference_graph_513_os16.pb"
input_tensor={"ImageTensor":[1,size,size,3]}
# Integer Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_513
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_slim_513_os16_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - edgetpu_deeplab_slim_513_os16_integer_quant.tflite")


size = 513
graph_def_file="frozen_inference_graph_513_os32.pb"
input_tensor={"ImageTensor":[1,size,size,3]}
# Integer Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_513
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_slim_513_os32_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - edgetpu_deeplab_slim_513_os32_integer_quant.tflite")


size = 769
graph_def_file="frozen_inference_graph_769_os16.pb"
input_tensor={"ImageTensor":[1,size,size,3]}
# Integer Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_769
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_slim_769_os16_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - edgetpu_deeplab_slim_769_os16_integer_quant.tflite")


size = 769
graph_def_file="frozen_inference_graph_769_os32.pb"
input_tensor={"ImageTensor":[1,size,size,3]}
# Integer Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_769
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_slim_769_os32_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - edgetpu_deeplab_slim_769_os32_integer_quant.tflite")