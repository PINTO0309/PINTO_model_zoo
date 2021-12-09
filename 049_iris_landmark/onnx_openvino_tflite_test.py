import onnxruntime
import tensorflow as tf
import time
import numpy as np
from pprint import pprint

H=64
W=64

interpreter = tf.lite.Interpreter(model_path=f'saved_model_{H}x{W}/iris_landmark.tflite', num_threads=4)
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
    result1 = interpreter.get_tensor(output_details[0]['index'])
    result2 = interpreter.get_tensor(output_details[1]['index'])
    e += (time.time() - s)
print('[ORIGINAL] tflite float32 output @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(f'elapsed time: {e/roop*1000}ms')
print(f'shape: {result2.shape}')
pprint(result2)
print(f'shape: {result1.shape}')
pprint(result1)
print('')
############################################################

onnx_session = onnxruntime.InferenceSession(f'saved_model_{H}x{W}/model_float32.onnx')
input_name = onnx_session.get_inputs()[0].name
output_name1 = onnx_session.get_outputs()[0].name
output_name2 = onnx_session.get_outputs()[1].name

roop = 1
e = 0.0
result = None
inp = np.ones((1,3,H,W), dtype=np.float32)
for _ in range(roop):
    s = time.time()
    result = onnx_session.run(
        [
            output_name1,
            output_name2
        ],
        {input_name: inp}
    )
    e += (time.time() - s)
print('ONNX output @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(f'elapsed time: {e/roop*1000}ms')
print(f'shape: {result[1].shape}')
pprint(result[1])
print(f'shape: {result[0].shape}')
pprint(result[0])
print('')
############################################################

from openvino.inference_engine import IECore
ie = IECore()
net = ie.read_network(
    model=f'saved_model_{H}x{W}/openvino/FP32/saved_model.xml',
    weights=f'saved_model_{H}x{W}/openvino/FP32/saved_model.bin'
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
print(f'shape: {result["output_iris"].shape}')
pprint(result['output_iris'])
print(f'shape: {result["output_eyes_contours_and_brows"].shape}')
pprint(result['output_eyes_contours_and_brows'])
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
    result1 = interpreter.get_tensor(output_details[0]['index'])
    result2 = interpreter.get_tensor(output_details[1]['index'])
    e += (time.time() - s)
print('tflite float32 output @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(f'elapsed time: {e/roop*1000}ms')
print(f'shape: {result1.shape}')
pprint(result1)
print(f'shape: {result2.shape}')
pprint(result2)
print('')
############################################################

interpreter = tf.lite.Interpreter(model_path=f'saved_model_{H}x{W}/model_integer_quant.tflite', num_threads=4)
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
    result1 = interpreter.get_tensor(output_details[0]['index'])
    result2 = interpreter.get_tensor(output_details[1]['index'])
    e += (time.time() - s)
print('tflite INT8 output @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(f'elapsed time: {e/roop*1000}ms')
print(f'shape: {result1.shape}')
pprint(result1)
print(f'shape: {result2.shape}')
pprint(result2)
