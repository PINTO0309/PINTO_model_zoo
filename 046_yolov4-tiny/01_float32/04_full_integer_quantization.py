### tensorflow==2.3.0-rc1

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def representative_dataset_gen():
    for data in raw_test_data.take(100):
        image = data['image'].numpy()
        image = tf.image.resize(image, (416, 416))
        image = image - 127.5
        image = image * 0.007843
        image = image[np.newaxis, :, :, :]
        yield [image]

raw_test_data, info = tfds.load(name="voc/2007", with_info=True, split="validation", data_dir="~/TFDS", download=False)

# Full Integer Quantization - Input/Output=float32
# INPUT  = input_1 (float32, 1 x 416 x 416 x 3)
# OUTPUT = conv2d_18, conv2d_21
model = tf.keras.models.model_from_json(open('yolov4_tiny_voc.json').read(), custom_objects={'tf': tf})
model.load_weights('yolov4_tiny_voc.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('yolov4_tiny_voc_416x416_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - yolov4_tiny_voc_416x416_full_integer_quant.tflite")
