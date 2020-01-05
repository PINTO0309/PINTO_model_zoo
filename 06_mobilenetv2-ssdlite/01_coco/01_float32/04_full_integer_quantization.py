import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def representative_dataset_gen():
  for data in raw_test_data.take(100):
    image = data['image'].numpy()
    image = tf.image.resize(image, (300, 300))
    image = image[np.newaxis,:,:,:]
    yield [image]


tf.compat.v1.enable_eager_execution()

raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()
with open('./ssdlite_mobilenet_v2_coco_300_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - ssdlite_mobilenet_v2_coco_300_full_integer_quant.tflite")
