import onnxruntime
import tensorflow as tf
import time
import numpy as np
from pprint import pprint

H=112
W=112

############################################################

# onnx_session = onnxruntime.InferenceSession(f'face_recognizer_fast.onnx')
# input_name = onnx_session.get_inputs()[0].name
# output_name = onnx_session.get_outputs()[0].name

# roop = 1
# e = 0.0
# result = None
# inp = np.ones((1,3,H,W), dtype=np.float32)
# for _ in range(roop):
#     s = time.time()
#     result = onnx_session.run(
#         [output_name],
#         {input_name: inp}
#     )
#     e += (time.time() - s)
# print('ONNX output @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# print(f'elapsed time: {e/roop*1000}ms')
# print(f'shape: {result[0].shape}')
# pprint(result)

############################################################
from openvino.inference_engine import IECore
ie = IECore()
net = ie.read_network(
    model=f'saved_model/openvino/FP32/face_recognizer_fast.xml',
    weights=f'saved_model/openvino/FP32/face_recognizer_fast.bin'
)
input_blob = next(iter(net.input_info))
exec_net = ie.load_network(network=net, device_name='CPU')

roop = 1
e = 0.0
result = None
inp = np.ones((1,3,H,W), dtype=np.float32)
for _ in range(roop):
    s = time.time()
    result = exec_net.infer(inputs={input_blob: inp})
    e += (time.time() - s)
print('OpenVINO output @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(f'elapsed time: {e/roop*1000}ms')
print(f'shape: {result["fc1"].shape}')
pprint(result['fc1'])

############################################################

interpreter = tf.lite.Interpreter(model_path=f'saved_model/model_float32.tflite', num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

roop = 1
e = 0.0
result = None
inp = np.ones((1,H,W,3), dtype=np.float32)
for _ in range(roop):
    s = time.time()
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])
    e += (time.time() - s)
print('tflite output @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(f'elapsed time: {e/roop*1000}ms')
print(f'shape: {result.shape}')
pprint(result)

