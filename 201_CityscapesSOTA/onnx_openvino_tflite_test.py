import onnxruntime
import tensorflow as tf
import time
import numpy as np
from pprint import pprint

H=180
W=320

onnx_session = onnxruntime.InferenceSession(f'saved_model_{H}x{W}/paddleseg_cityscapessota_{H}x{W}.onnx')
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

roop = 1
e = 0.0
result = None
inp = np.ones((1,3,H,W), dtype=np.float32)
for _ in range(roop):
    s = time.time()
    result = onnx_session.run(
        [
            output_name,
        ],
        {input_name: inp}
    )
    e += (time.time() - s)
print('[Original] ONNX output @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(f'elapsed time: {e/roop*1000}ms')
print(f'shape: {result[0].shape}')
pprint(result[0])
print('')
############################################################

onnx_session = onnxruntime.InferenceSession(f'saved_model_{H}x{W}/model_float32.onnx')
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

roop = 1
e = 0.0
result = None
inp = np.ones((1,3,H,W), dtype=np.float32)
for _ in range(roop):
    s = time.time()
    result = onnx_session.run(
        [
            output_name,
        ],
        {input_name: inp}
    )
    e += (time.time() - s)
print('[Optimized] ONNX output @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(f'elapsed time: {e/roop*1000}ms')
print(f'shape: {result[0].transpose((0,3,1,2)).shape}')
pprint(result[0].transpose((0,3,1,2)))
print('')
############################################################

from openvino.inference_engine import IECore
ie = IECore()
net = ie.read_network(
    model=f'saved_model_{H}x{W}/openvino/FP32/paddleseg_cityscapessota_{H}x{W}.xml',
    weights=f'saved_model_{H}x{W}/openvino/FP32/paddleseg_cityscapessota_{H}x{W}.bin'
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
print(f'shape: {result["tmp_520"].shape}')
pprint(result['tmp_520'])
print('')
############################################################

interpreter = tf.lite.Interpreter(model_path=f'saved_model_{H}x{W}/model_float32.tflite', num_threads=4)
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
print('tflite float32 output @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(f'elapsed time: {e/roop*1000}ms')
print(f'shape: {result.transpose((0,3,1,2)).shape}')
pprint(result.transpose((0,3,1,2)))
print('')
############################################################

# interpreter = tf.lite.Interpreter(model_path=f'saved_model_{H}x{W}/model_integer_quant.tflite', num_threads=4)
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# roop = 1
# e = 0.0
# result = None
# inp = np.ones((1,H,W,3), dtype=np.float32)
# for _ in range(roop):
#     s = time.time()
#     interpreter.set_tensor(input_details[0]['index'], inp)
#     interpreter.invoke()
#     result = interpreter.get_tensor(output_details[0]['index'])
#     e += (time.time() - s)
# print('tflite INT8 output @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# print(f'elapsed time: {e/roop*1000}ms')
# print(f'shape: {result.shape}')
# pprint(result)
