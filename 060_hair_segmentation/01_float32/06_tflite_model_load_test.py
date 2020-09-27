import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="hair_segmentation_512x512_float32.tflite")
# interpreter = tf.lite.Interpreter(model_path="hair_segmentation_512x512_weight_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print('input:', input_details)
print('')
print('output:', output_details)
 
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print('output_data.shape:', output_data.shape)

import cv2
