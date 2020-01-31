import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def representative_dataset_gen():
  for data in raw_test_data.take(100):
    image = data['image'].numpy()
    image = tf.image.resize(image, (300, 300))
    image = image[np.newaxis,:,:,:]
    image = image - 127.5
    image = image * 0.007843
    yield [image]


tf.compat.v1.enable_eager_execution()

raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)

# Integer Quantization - Input/Output=float32
graph_def_file="tflite_graph.pb"
input_arrays=["normalized_input_image_tensor"]
output_arrays=['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']
input_tensor={"normalized_input_image_tensor":[1,300,300,3]}

converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)
converter.allow_custom_ops=True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('./ssdlite_mobilenet_v2_coco_300_integer_quant_with_postprocess.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - ssdlite_mobilenet_v2_coco_300_integer_quant_with_postprocess.tflite")
