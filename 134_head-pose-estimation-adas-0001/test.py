import numpy as np
import cv2
from tensorflow.lite.python.interpreter import Interpreter
from pprint import pprint

interpreter = Interpreter(model_path='saved_model/model_float32.tflite', num_threads=4)
interpreter.allocate_tensors()
input_blob = interpreter.get_input_details()
output_blob = interpreter.get_output_details()

"""
[{'dtype': <class 'numpy.float32'>,
  'index': 0,
  'name': 'data',
  'quantization': (0.0, 0),
  'quantization_parameters': {'quantized_dimension': 0,
                              'scales': array([], dtype=float32),
                              'zero_points': array([], dtype=int32)},
  'shape': array([ 1, 60, 60,  3], dtype=int32),
  'shape_signature': array([ 1, 60, 60,  3], dtype=int32),
  'sparsity_parameters': {}}]
"""
pprint(input_blob)
pprint(output_blob)

img = cv2.imread("test.jpg")
img = cv2.resize(img, (60, 60))
img = img.astype(np.float32)
img = img[np.newaxis, :, :, :]

interpreter.set_tensor(input_blob[0]['index'], img)
interpreter.invoke()

out0 = interpreter.get_tensor(output_blob[0]['index'])
out1 = interpreter.get_tensor(output_blob[1]['index'])
out2 = interpreter.get_tensor(output_blob[2]['index'])

"""
[{'dtype': <class 'numpy.float32'>,
  'index': 85,
  'name': 'Identity',
  'quantization': (0.0, 0),
  'quantization_parameters': {'quantized_dimension': 0,
                              'scales': array([], dtype=float32),
                              'zero_points': array([], dtype=int32)},
  'shape': array([1, 1], dtype=int32),
  'shape_signature': array([1, 1], dtype=int32),
  'sparsity_parameters': {}},
 {'dtype': <class 'numpy.float32'>,
  'index': 88,
  'name': 'Identity_1',
  'quantization': (0.0, 0),
  'quantization_parameters': {'quantized_dimension': 0,
                              'scales': array([], dtype=float32),
                              'zero_points': array([], dtype=int32)},
  'shape': array([1, 1], dtype=int32),
  'shape_signature': array([1, 1], dtype=int32),
  'sparsity_parameters': {}},
 {'dtype': <class 'numpy.float32'>,
  'index': 80,
  'name': 'Identity_2',
  'quantization': (0.0, 0),
  'quantization_parameters': {'quantized_dimension': 0,
                              'scales': array([], dtype=float32),
                              'zero_points': array([], dtype=int32)},
  'shape': array([1, 1], dtype=int32),
  'shape_signature': array([1, 1], dtype=int32),
  'sparsity_parameters': {}}]
"""
print(f'fc_p: {np.sum(out2)}')
print(f'fc_r: {np.sum(out1)}')
print(f'fc_y: {np.sum(out0)}')

from openvino.inference_engine import IENetwork, IECore

ie = IECore()
model='head-pose-estimation-adas-0001'
net = ie.read_network(f'{model}.xml', f'{model}.bin')
input_blob = next(iter(net.input_info))
out_blob   = next(iter(net.outputs))
exec_net = ie.load_network(net, 'CPU')

"""
'data'
"""
pprint(input_blob)
"""
{'angle_p_fc': <openvino.inference_engine.ie_api.DataPtr object at 0x7fa6e5eb9530>,
 'angle_r_fc': <openvino.inference_engine.ie_api.DataPtr object at 0x7fa6e32c74b0>,
 'angle_y_fc': <openvino.inference_engine.ie_api.DataPtr object at 0x7fa6e32c7430>}
"""
pprint(net.outputs)

res = exec_net.infer(inputs={input_blob: img.transpose(0,3,1,2)})
pprint(res)
