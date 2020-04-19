### tf-nightly-2.2.0.dev20200418

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('./yolov3_nano_voc_416_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - yolov3_nano_voc_416_float16_quant.tflite")
