### tensorflow==2.3.0-rc1

import tensorflow as tf

# tf.compat.v1.enable_eager_execution()

# Float16 Quantization - Input/Output=float32
# INPUT  = input_1 (float32, 1 x 416 x 416 x 3)
# OUTPUT = conv2d_18, conv2d_21
model = tf.keras.models.model_from_json(open('yolov4_tiny_voc.json').read(), custom_objects={'tf': tf})
model.load_weights('yolov4_tiny_voc.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('yolov4_tiny_voc_416x416_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - yolov4_tiny_voc_416x416_float16_quant.tflite")