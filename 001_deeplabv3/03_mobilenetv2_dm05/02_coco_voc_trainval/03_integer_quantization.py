import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def representative_dataset_gen():
  for data in raw_test_data.take(10):
    image = data['image'].numpy()
    image = tf.image.resize(image, (513, 513))
    image = image[np.newaxis,:,:,:]
    yield [image]


tf.compat.v1.enable_eager_execution()

raw_test_data, info = tfds.load(name="voc/2007", with_info=True, split="validation", data_dir="~/TFDS", download=False)

graph_def_file="frozen_inference_graph.pb"
input_arrays=["ImageTensor"]
output_arrays=['ArgMax']
input_tensor={"ImageTensor":[1,513,513,3]}

# Integer Quantization - Input/Output=float32
#converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays,input_tensor)

converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('./deeplabv3_mnv2_pascal_trainval_513_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - deeplabv3_mnv2_pascal_trainval_513_integer_quant.tflite")
