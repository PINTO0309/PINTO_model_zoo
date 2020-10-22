### tensorflow==2.3.1

import tensorflow.compat.v1 as tf

converter = tf.lite.TFLiteConverter.from_frozen_graph(
  'model_float32.pb', ['inputs'], ['Identity'])
tflite_model = converter.convert()
open("model_float32.tflite", "wb").write(tflite_model)