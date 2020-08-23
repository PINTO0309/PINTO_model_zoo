### tensorflow==2.3.0-rc1

import tensorflow as tf

# tf.compat.v1.enable_eager_execution()

# Weight Quantization - Input/Output=float32
# INPUT  = input_1 (float32, 1 x 416 x 416 x 3)
# OUTPUT = conv2d_18, conv2d_21
model = tf.keras.models.model_from_json(open('yolov4_tiny_voc.json').read(), custom_objects={'tf': tf})
model.load_weights('yolov4_tiny_voc.h5')
model.save('saved_model')