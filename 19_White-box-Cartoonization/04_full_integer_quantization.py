### tensorflow==2.1.0 v1-API

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def representative_dataset_gen():
  for data in raw_test_data.take(100):
    image = data['image'].numpy()
    image = tf.image.resize(image, (720, 720))
    image = image / 127.5 - 1
    image = image[np.newaxis,:,:,:]
    yield [image]


tf.compat.v1.enable_eager_execution()

raw_test_data, info = tfds.load(name="voc/2007", with_info=True, split="validation", data_dir="~/TFDS", download=False)

# Integer Quantization - Input/Output=float32
input_arrays=["input"]
output_arrays=['add_1']
size = 720
graph_def_file="export/white_box_cartoonization_freeze_graph.pb"
input_tensor={"input":[1,size,size,3]}
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('export/white_box_cartoonization_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - white_box_cartoonization_full_integer_quant.tflite")
