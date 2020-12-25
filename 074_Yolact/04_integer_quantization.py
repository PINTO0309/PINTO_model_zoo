### tensorflow==2.5.0-dev20201204

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

height = 550
width  = 550
MEANS = (103.94, 116.78, 123.68)
STD   = (57.38, 57.12, 58.40)

def representative_dataset_gen():
    for image in raw_test_data.take(10):
        image = image['image'].numpy()
        image = tf.image.resize(image, (height, width))
        image = image[np.newaxis,:,:,:]
        image = image / 255.0
        yield [image]

raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)

# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_from_pb')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.representative_dataset = representative_dataset_gen
tflite_model = converter.convert()
with open('yolact_550x550_opt_integer_quant.tflite', 'wb') as w:
    w.write(tflite_model)
print('Integer Quantization complete! - yolact_550x550_opt_integer_quant.tflite')
