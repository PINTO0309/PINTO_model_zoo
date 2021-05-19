import numpy as np
import time
import tensorflow.lite as tflite
import cv2
import sys

interpreter = tflite.Interpreter(model_path='model_float32.tflite', num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = cv2.cvtColor(cv2.imread('test.png'), cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256))
image = np.expand_dims(image, axis=0)
image = image.astype('float32')

interpreter.set_tensor(input_details[0]['index'], image)
start_time = time.time()
interpreter.invoke()
stop_time = time.time()
print("time: ", stop_time - start_time)

scores = interpreter.get_tensor(output_details[0]['index'])

import pprint
pprint.pprint(scores)

