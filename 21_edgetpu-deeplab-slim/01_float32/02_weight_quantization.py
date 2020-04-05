import tensorflow as tf

### Tensorflow v1.15.2

#tf.compat.v1.enable_eager_execution()

graph_def_file="frozen_inference_graph_257_os16.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,257,257,3]}
# Weight Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
#converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_slim_257_os16_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - edgetpu_deeplab_slim_257_os16_weight_quant.tflite")



graph_def_file="frozen_inference_graph_257_os32.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,257,257,3]}
# Weight Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
#converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_slim_257_os32_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - edgetpu_deeplab_slim_257_os32_weight_quant.tflite")



graph_def_file="frozen_inference_graph_321_os16.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,321,321,3]}
# Weight Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
#converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_slim_321_os16_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - edgetpu_deeplab_slim_321_os16_weight_quant.tflite")



graph_def_file="frozen_inference_graph_321_os32.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,321,321,3]}
# Weight Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
#converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_slim_321_os32_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - edgetpu_deeplab_slim_321_os32_weight_quant.tflite")



graph_def_file="frozen_inference_graph_513_os16.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,513,513,3]}
# Weight Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
#converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_slim_513_os16_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - edgetpu_deeplab_slim_513_os16_weight_quant.tflite")



graph_def_file="frozen_inference_graph_513_os32.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,513,513,3]}
# Weight Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
#converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_slim_513_os32_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - edgetpu_deeplab_slim_513_os32_weight_quant.tflite")



graph_def_file="frozen_inference_graph_769_os16.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,769,769,3]}
# Weight Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
#converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_slim_769_os16_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - edgetpu_deeplab_slim_769_os16_weight_quant.tflite")



graph_def_file="frozen_inference_graph_769_os32.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,769,769,3]}
# Weight Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
#converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./edgetpu_deeplab_slim_769_os32_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - edgetpu_deeplab_slim_769_os32_weight_quant.tflite")