### tf-nightly==2.3.0-dev20200527

import tensorflow as tf

# Weight Quantization - Input/Output=float32
graph_def_file="tflite_graph_with_postprocess.pb"
input_arrays=["normalized_input_image_tensor"]
output_arrays=['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']
input_tensor={"normalized_input_image_tensor":[1,320,320,3]}
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.allow_custom_ops=True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('ssdlite_mobiledet_cpu_320x320_coco_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - ssdlite_mobiledet_cpu_320x320_coco_weight_quant.tflite")