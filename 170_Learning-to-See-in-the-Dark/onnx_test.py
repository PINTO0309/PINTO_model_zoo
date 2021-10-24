"""
docker run --rm -it --gpus all \
-v $PWD:/workspace/work \
pinto0309/cuda114-tensorrt82 /bin/bash

cd work
python3 onnx_test.py
"""

import onnxruntime
import time
import numpy as np

onnx_session = onnxruntime.InferenceSession('saved_model_sony_480x640/model_float32.onnx')
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

roop = 20
e = 0.0
inp = np.ones((1,4,480,640), dtype=np.float32)
for _ in range(roop):
    s = time.time()
    result = onnx_session.run(
        [output_name],
        {input_name: inp}
    )
    e += (time.time() - s)
print(f'elapsed time: {e/roop*1000}ms')
"""
elapsed time: 57.117438316345215ms
"""


import onnx
import onnx_tensorrt.backend as be

model = onnx.load('saved_model_sony_480x640/model_float32.onnx')
engine = be.prepare(model, device='CUDA:0')
e = 0.0
for _ in range(roop):
    s = time.time()
    output = engine.run(inp)[0]
    e += (time.time() - s)
print(f'elapsed time: {e/roop*1000}ms')
"""
elapsed time: 13.761746883392334ms
"""
