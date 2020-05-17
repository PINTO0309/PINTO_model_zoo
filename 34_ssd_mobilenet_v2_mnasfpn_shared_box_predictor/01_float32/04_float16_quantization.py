import tensorflow as tf

tf.compat.v1.enable_eager_execution()

# Float16 Quantization - Input/Output=float32
graph_def_file="tflite_graph_with_postprocess.pb"
input_arrays=["normalized_input_image_tensor"]
output_arrays=['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1', 
               'TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']
input_tensor={"normalized_input_image_tensor":[1,320,320,3]}

converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, 
                                                      output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.allow_custom_ops = True
tflite_quant_model = converter.convert()
with open('./ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320_coco_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320_coco_float16_quant.tflite")