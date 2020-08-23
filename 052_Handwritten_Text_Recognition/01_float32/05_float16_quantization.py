### tensorflow==2.3.0

import tensorflow.compat.v1 as tf

# Float16 Quantization - Input/Output=float32
graph_def_file="simpleHTR_freeze_graph_opt.pb"
input_arrays=["input"]
output_arrays=['CTCGreedyDecoder','CTCGreedyDecoder:1','CTCGreedyDecoder:2','CTCGreedyDecoder:3']
input_tensor={"input":[1,128,32,1]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('simpleHTR_128x32_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - simpleHTR_128x32_float16_quant.tflite")
