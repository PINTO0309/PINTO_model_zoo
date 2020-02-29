import tensorflow as tf

### Tensorflow v2.1.0 - master - commit 8fdb834931fe62abaeab39fe6bf0bcc9499b25bf

#tf.compat.v1.enable_eager_execution()

graph_def_file="frozen_inference_graph_257.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,257,257,3]}

# Integer Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)

#converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
with open('./deeplabv3_mnv2_pascal_trainval_257_weight_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - deeplabv3_mnv2_pascal_trainval_257_weight_quant.tflite")
