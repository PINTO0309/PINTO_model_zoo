### tf_nightly-2.5.0.dev20201123

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

mean = 0.0
std = 1.0

height = 60
width  = 60

def representative_dataset_gen():
    for data in raw_test_data.take(10):
        image = data['image'].numpy()
        image = tf.image.resize(image, (60, 60))
        image = (image / 255 - mean) / std
        image = image[np.newaxis,:,:,:]

        angles = tf.random.uniform([1,3], minval=0.0, maxval=7.0)
        
        yield [image,image,angles]

raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_for_edgetpu')
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.representative_dataset = representative_dataset_gen
tflite_model = converter.convert()
with open('saved_model_for_edgetpu/model_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_model)
print('Integer Quantization complete! - saved_model_for_edgetpu/model_full_integer_quant.tflite')
