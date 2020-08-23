### tf-nightly-2.2.0.dev20200410

import tensorflow as tf

# Weight Quantization - Input/Output=float32
# graph_def_file="1_checkpoint/16_bit_HE_to_HE_gt/lsmod_256.pb"
# input_arrays=["input"]
# output_arrays=['output']
# input_tensor={"input":[1,16,256,256,4]}
#converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)

converter = tf.lite.TFLiteConverter.from_saved_model('./3_saved_model')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
tflite_quant_model = converter.convert()
with open('1_checkpoint/16_bit_HE_to_HE_gt/lsmod_256_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Weight Quantization complete! - lsmod_256_weight_quant.tflite")

