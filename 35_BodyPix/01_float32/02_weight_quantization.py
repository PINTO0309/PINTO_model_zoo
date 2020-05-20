### tensorflow-gpu==1.15.2

import tensorflow as tf
import numpy as np

# Weight Quantization - Input/Output=float32
graph_def_file="bodypix_025.pb"
input_arrays=["sub"]
output_arrays=['output_raw_heatmaps','output_raw_offsets', 
               'output_raw_part_heatmaps/conv','output_raw_segments']
input_tensor={"sub":[1,640,480,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('bodypix_025_640x480_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - bodypix_025_640x480_weight_quant.tflite")


# Weight Quantization - Input/Output=float32
graph_def_file="bodypix_025.pb"
input_arrays=["sub"]
output_arrays=['output_raw_heatmaps','output_raw_offsets', 
               'output_raw_part_heatmaps/conv','output_raw_segments']
input_tensor={"sub":[1,320,240,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('bodypix_025_320x240_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - bodypix_025_320x240_weight_quant.tflite")


# Weight Quantization - Input/Output=float32
graph_def_file="bodypix_050.pb"
input_arrays=["sub"]
output_arrays=['output_raw_heatmaps','output_raw_offsets', 
               'output_raw_part_heatmaps/conv','output_raw_segments']
input_tensor={"sub":[1,640,480,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('bodypix_050_640x480_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - bodypix_050_640x480_weight_quant.tflite")


# Weight Quantization - Input/Output=float32
graph_def_file="bodypix_050.pb"
input_arrays=["sub"]
output_arrays=['output_raw_heatmaps','output_raw_offsets', 
               'output_raw_part_heatmaps/conv','output_raw_segments']
input_tensor={"sub":[1,320,240,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('bodypix_050_320x240_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - bodypix_050_320x240_weight_quant.tflite")


# Weight Quantization - Input/Output=float32
graph_def_file="bodypix_075.pb"
input_arrays=["sub"]
output_arrays=['output_raw_heatmaps','output_raw_offsets', 
               'output_raw_part_heatmaps/conv','output_raw_segments']
input_tensor={"sub":[1,640,480,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('bodypix_075_640x480_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - bodypix_075_640x480_weight_quant.tflite")


# Weight Quantization - Input/Output=float32
graph_def_file="bodypix_075.pb"
input_arrays=["sub"]
output_arrays=['output_raw_heatmaps','output_raw_offsets', 
               'output_raw_part_heatmaps/conv','output_raw_segments']
input_tensor={"sub":[1,320,240,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('bodypix_075_320x240_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - bodypix_075_320x240_weight_quant.tflite")


# Weight Quantization - Input/Output=float32
graph_def_file="bodypix_100.pb"
input_arrays=["sub"]
output_arrays=['output_raw_heatmaps','output_raw_offsets', 
               'output_raw_part_heatmaps/conv','output_raw_segments']
input_tensor={"sub":[1,640,480,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('bodypix_100_640x480_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - bodypix_100_640x480_weight_quant.tflite")


# Weight Quantization - Input/Output=float32
graph_def_file="bodypix_100.pb"
input_arrays=["sub"]
output_arrays=['output_raw_heatmaps','output_raw_offsets', 
               'output_raw_part_heatmaps/conv','output_raw_segments']
input_tensor={"sub":[1,320,240,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('bodypix_100_320x240_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - bodypix_100_320x240_weight_quant.tflite")