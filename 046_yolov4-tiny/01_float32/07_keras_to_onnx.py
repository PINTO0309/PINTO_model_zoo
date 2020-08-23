### tensorflow==2.3.0-rc1

import tensorflow as tf
import keras2onnx

# Weight Quantization - Input/Output=float32
# INPUT  = input_1 (float32, 1 x 416 x 416 x 3)
# OUTPUT = conv2d_18, conv2d_21
model = tf.keras.models.model_from_json(open('yolov4_tiny_voc.json').read(), custom_objects={'tf': tf})
model.load_weights('yolov4_tiny_voc.h5')
onnx_model = keras2onnx.convert_keras(model, model.name, channel_first_inputs=['input_1'])
keras2onnx.save_model(onnx_model, 'yolov4_tiny_voc_416x416.onnx')