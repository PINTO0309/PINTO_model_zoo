### tf-nightly==2.3.0-dev20200527

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Float16 Quantization - Input/Output=float32
graph_def_file="tflite_graph_with_postprocess.pb"
input_arrays=["normalized_input_image_tensor"]
output_arrays=['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']
input_tensor={"normalized_input_image_tensor":[1,320,320,3]}
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.allow_custom_ops=True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('ssdlite_mobiledet_edgetpu_320x320_coco_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - ssdlite_mobiledet_edgetpu_320x320_coco_float16_quant.tflite")