import onnx
import onnxruntime
from onnxsim import simplify
import argparse
import os
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--height", type=int, required=True)
parser.add_argument("--width", type=int, required=True)
args = parser.parse_args()

H=args.height
W=args.width
MODEL=args.model
model = onnx.load(f'{MODEL}')

onnx_session = onnxruntime.InferenceSession(f'{MODEL}')
inputs = {}

for i in onnx_session.get_inputs():
    inputs[i.name] = [i.shape[0], i.shape[1], H, W]

print('@@@@@@@@@@@@@@@@@@@@@ inputs')
pprint(inputs)

model_simp, check = simplify(
    model,
    input_shapes=inputs
)
basename_without_ext = \
    os.path.splitext(os.path.basename(MODEL))[0].split('_')[0]  + "_" \
    + os.path.splitext(os.path.basename(MODEL))[0].split('_')[1] + "_" \
    + os.path.splitext(os.path.basename(MODEL))[0].split('_')[2] + "_" \
    + os.path.splitext(os.path.basename(MODEL))[0].split('_')[3] + "_" \
    + os.path.splitext(os.path.basename(MODEL))[0].split('_')[4]
onnx.save(model_simp, f'{basename_without_ext}_{H}x{W}.onnx')