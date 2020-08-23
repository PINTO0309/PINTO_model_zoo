import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def representative_dataset_gen_640():
  for data in raw_test_data.take(100):
    image = data['image'].numpy()
    image = tf.image.resize(image, (640, 480))
    image = image[np.newaxis,:,:,:]
    image = image - 127.5
    image = image * 0.007843
    yield [image]

def representative_dataset_gen_320():
  for data in raw_test_data.take(100):
    image = data['image'].numpy()
    image = tf.image.resize(image, (320, 240))
    image = image[np.newaxis,:,:,:]
    image = image - 127.5
    image = image * 0.007843
    yield [image]

tf.compat.v1.enable_eager_execution()

raw_test_data, info = tfds.load(name="voc/2007", with_info=True, 
                                split="validation", data_dir="~/TFDS", download=False)

# Integer Quantization - Input/Output=float32
graph_def_file="bodypix_025.pb"
input_arrays=["sub"]
output_arrays=['output_raw_heatmaps','output_raw_offsets', 
               'output_raw_part_heatmaps/conv','output_raw_segments']
input_tensor={"sub":[1,640,480,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_640
tflite_quant_model = converter.convert()
with open('bodypix_025_640x480_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - bodypix_025_640x480_integer_quant.tflite")


# Integer Quantization - Input/Output=float32
graph_def_file="bodypix_025.pb"
input_arrays=["sub"]
output_arrays=['output_raw_heatmaps','output_raw_offsets', 
               'output_raw_part_heatmaps/conv','output_raw_segments']
input_tensor={"sub":[1,320,240,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_320
tflite_quant_model = converter.convert()
with open('bodypix_025_320x240_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - bodypix_025_320x240_integer_quant.tflite")


# Integer Quantization - Input/Output=float32
graph_def_file="bodypix_050.pb"
input_arrays=["sub"]
output_arrays=['output_raw_heatmaps','output_raw_offsets', 
               'output_raw_part_heatmaps/conv','output_raw_segments']
input_tensor={"sub":[1,640,480,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_640
tflite_quant_model = converter.convert()
with open('bodypix_050_640x480_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - bodypix_050_640x480_integer_quant.tflite")


# Integer Quantization - Input/Output=float32
graph_def_file="bodypix_050.pb"
input_arrays=["sub"]
output_arrays=['output_raw_heatmaps','output_raw_offsets', 
               'output_raw_part_heatmaps/conv','output_raw_segments']
input_tensor={"sub":[1,320,240,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_320
tflite_quant_model = converter.convert()
with open('bodypix_050_320x240_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - bodypix_050_320x240_integer_quant.tflite")


# Integer Quantization - Input/Output=float32
graph_def_file="bodypix_075.pb"
input_arrays=["sub"]
output_arrays=['output_raw_heatmaps','output_raw_offsets', 
               'output_raw_part_heatmaps/conv','output_raw_segments']
input_tensor={"sub":[1,640,480,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_640
tflite_quant_model = converter.convert()
with open('bodypix_075_640x480_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - bodypix_075_640x480_integer_quant.tflite")


# Integer Quantization - Input/Output=float32
graph_def_file="bodypix_075.pb"
input_arrays=["sub"]
output_arrays=['output_raw_heatmaps','output_raw_offsets', 
               'output_raw_part_heatmaps/conv','output_raw_segments']
input_tensor={"sub":[1,320,240,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_320
tflite_quant_model = converter.convert()
with open('bodypix_075_320x240_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - bodypix_075_320x240_integer_quant.tflite")


# Integer Quantization - Input/Output=float32
graph_def_file="bodypix_100.pb"
input_arrays=["sub"]
output_arrays=['output_raw_heatmaps','output_raw_offsets', 
               'output_raw_part_heatmaps/conv','output_raw_segments']
input_tensor={"sub":[1,640,480,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_640
tflite_quant_model = converter.convert()
with open('bodypix_100_640x480_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - bodypix_100_640x480_integer_quant.tflite")


# Integer Quantization - Input/Output=float32
graph_def_file="bodypix_100.pb"
input_arrays=["sub"]
output_arrays=['output_raw_heatmaps','output_raw_offsets', 
               'output_raw_part_heatmaps/conv','output_raw_segments']
input_tensor={"sub":[1,320,240,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen_320
tflite_quant_model = converter.convert()
with open('bodypix_100_320x240_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - bodypix_100_320x240_integer_quant.tflite")