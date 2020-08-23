### tf_nightly-2.2.0.dev20200408

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import cv2

def representative_dataset_gen():
  for data in raw_test_data.take(100):
    image = data['image'].numpy()
    image = tf.image.resize(image, (256, 256))
    image = image[np.newaxis,:,:,:]
    yield [image]


#tf.compat.v1.enable_eager_execution()

raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)

# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_spec.supported_types = [tf.float16]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('./mask_rcnn_inception_v2_coco_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - mask_rcnn_inception_v2_coco_float16_quant.tflite")