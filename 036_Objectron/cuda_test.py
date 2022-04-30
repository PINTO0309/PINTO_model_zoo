import onnxruntime
import numpy as np
from pprint import pprint

### Batch N test
BATCH=5
# ONNX
onnx_session = onnxruntime.InferenceSession(
    'model_float32_camera_Nx224x224.onnx',
    providers=[
        'CUDAExecutionProvider',
    ],
)
# Inference
input_name = onnx_session.get_inputs()[0].name
results = onnx_session.run(
    None,
    {input_name: np.ones([BATCH,3,224,224], dtype=np.float32)}
)
for result in results:
    pprint(result)
