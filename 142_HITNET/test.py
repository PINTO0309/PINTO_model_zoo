import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
import numpy as np
from pprint import pprint
import time
import platform

H=256
W=256

THREADS=4

# MODEL='flyingthings_finalpass_xl'
# CHANNEL=6
MODEL='eth3d'
CHANNEL=2
# MODEL='middlebury_d400'
# CHANNEL=6

interpreter = tf.lite.Interpreter(f'{MODEL}/saved_model_{H}x{W}/model_float32.tflite', num_threads=THREADS)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_height = input_shape[1]
input_width = input_shape[2]
channels = input_shape[3]

size = (1, input_height, input_width, CHANNEL)
input_tensor = np.ones(size, dtype=np.float32)

start = time.perf_counter()
roop_count = 10
reference_output_disparity = None
for i in range(roop_count):
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    reference_output_disparity = interpreter.get_tensor(output_details[0]['index'])

inference_time = (time.perf_counter() - start) / roop_count
# pprint(reference_output_disparity)
print(f'Model: {MODEL}')
print(f'Input resolution: {H}x{W}')
print(f'Number of Threads: {THREADS}')
print(f'Platform: {platform.platform()}')
print(f'Average of {roop_count} times inference: {(inference_time * 1000):.1f}ms')

"""
$ python3 test.py
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
INFO: Created TensorFlow Lite delegate for select TF ops.
INFO: TfLiteFlexDelegate delegate: 20 nodes delegated out of 772 nodes with 10 partitions.

Model: eth3d
Input resolution: 256x256
Number of Threads: 4
Platform: Linux-5.11.0-27-generic-x86_64-with-glibc2.29
Average of 10 times inference: 360.6ms
"""