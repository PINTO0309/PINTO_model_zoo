import onnxruntime
import tensorflow as tf
import time
import numpy as np
from pprint import pprint

import torch

H=180
W=320
MODEL=f'lstr_{H}x{W}'

############################################################

onnx_session = onnxruntime.InferenceSession(f'lstr_{H}x{W}/{MODEL}.onnx')
input_name1 = onnx_session.get_inputs()[0].name
input_name2 = onnx_session.get_inputs()[1].name
output_names = [o.name for o in onnx_session.get_outputs()]

roop = 1
e = 0.0
result = None
inp1 = np.ones((1,3,H,W), dtype=np.float32)
inp2 = np.zeros((1,1,H,W), dtype=np.float32)
for _ in range(roop):
    s = time.time()
    result = onnx_session.run(
        output_names,
        {
            input_name1: inp1,
            input_name2: inp2,
        }
    )
    e += (time.time() - s)
print('ONNX output @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(f'elapsed time: {e/roop*1000}ms')
print(f'shape: {result[0].shape}')
pprint(result)

############################################################

interpreter = tf.lite.Interpreter(model_path=f'lstr_{H}x{W}/model_float32.tflite', num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

roop = 1
e = 0.0
result = None
inp1 = np.ones((1,H,W,3), dtype=np.float32)
inp2 = np.zeros((1,H,W,1), dtype=np.float32)
for _ in range(roop):
    s = time.time()
    interpreter.set_tensor(input_details[0]['index'], inp1)
    interpreter.set_tensor(input_details[1]['index'], inp2)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])
    e += (time.time() - s)
print('tflite output @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(f'elapsed time: {e/roop*1000}ms')
print(f'shape: {result.shape}')
pprint(result)
