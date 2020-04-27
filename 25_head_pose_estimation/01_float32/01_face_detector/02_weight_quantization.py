### tensorflow-gpu==1.15.2

import tensorflow as tf

tf.compat.v1.enable_eager_execution()

# Weight Quantization - Input/Output=float32
graph_def_file="export/tflite_graph.pb"
input_arrays=["normalized_input_image_tensor"]
output_arrays=['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']
input_tensor={"normalized_input_image_tensor":[1,300,300,3]}

converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.allow_custom_ops=True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./ssdlite_mobilenet_v2_face_300_weight_quant_with_postprocess.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - ssdlite_mobilenet_v2_face_300_weight_quant_with_postprocess.tflite")

