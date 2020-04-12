### Tensorflow v1.15.2

import tensorflow as tf

# Float16 Quantization - Input/Output=float32
graph_def_file="frozen_inference_graph_257_os16.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,257,257,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_257_os16_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - edgetpu_deeplab_257_os16_float16_quant.tflite")

# Float16 Quantization - Input/Output=float32
graph_def_file="frozen_inference_graph_257_os32.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,257,257,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_257_os32_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - edgetpu_deeplab_257_os32_float16_quant.tflite")

# Float16 Quantization - Input/Output=float32
graph_def_file="frozen_inference_graph_321_os16.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,321,321,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_321_os16_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - edgetpu_deeplab_321_os16_float16_quant.tflite")

# Float16 Quantization - Input/Output=float32
graph_def_file="frozen_inference_graph_321_os32.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,321,321,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_321_os32_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - edgetpu_deeplab_321_os32_float16_quant.tflite")

# Float16 Quantization - Input/Output=float32
graph_def_file="frozen_inference_graph_513_os16.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,513,513,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_513_os16_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - edgetpu_deeplab_513_os16_float16_quant.tflite")

# Float16 Quantization - Input/Output=float32
graph_def_file="frozen_inference_graph_513_os32.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,513,513,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_513_os32_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - edgetpu_deeplab_513_os32_float16_quant.tflite")

# Float16 Quantization - Input/Output=float32
graph_def_file="frozen_inference_graph_769_os16.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,769,769,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_769_os16_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - edgetpu_deeplab_769_os16_float16_quant.tflite")

# Float16 Quantization - Input/Output=float32
graph_def_file="frozen_inference_graph_769_os32.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,769,769,3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_769_os32_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - edgetpu_deeplab_769_os32_float16_quant.tflite")