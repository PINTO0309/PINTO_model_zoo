import onnx
import onnxruntime
from onnxsim import simplify
import argparse
import os
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
args = parser.parse_args()

MODEL=args.model
BATCH_SIZE=args.batch_size
model = onnx.load(f'{MODEL}')

onnx_session = onnxruntime.InferenceSession(
    f'{MODEL}',
    providers=['CPUExecutionProvider']
)
inputs = {}

input_layer = onnx_session.get_inputs()[0]
input_shape_len = len(input_layer.shape)

if input_shape_len == 4:
    inputs[input_layer.name] = [
        BATCH_SIZE,
        input_layer.shape[1],
        input_layer.shape[2],
        input_layer.shape[3]
    ]

elif input_shape_len == 2:
    inputs[input_layer.name] = [
        BATCH_SIZE,
        input_layer.shape[1]
    ]

print('@@@@@@@@@@@@@@@@@@@@@ inputs')
pprint(inputs)

model_simp, check = simplify(
    model,
    input_shapes=inputs
)

if input_shape_len == 4:
    basename_without_ext = \
        os.path.splitext(os.path.basename(MODEL))[0].split('_')[0] \
        + "_" + os.path.splitext(os.path.basename(MODEL))[0].split('_')[1]
    onnx.save(model_simp, f'{basename_without_ext}_{BATCH_SIZE}x3x256x256.onnx')

elif input_shape_len == 2:
    basename_without_ext = \
        os.path.splitext(os.path.basename(MODEL))[0].split('_')[0] \
        + "_" + os.path.splitext(os.path.basename(MODEL))[0].split('_')[1]
    onnx.save(model_simp, f'{basename_without_ext}_{BATCH_SIZE}x66.onnx')