import onnx
from onnxsim import simplify
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--height", type=int, required=True)
parser.add_argument("--width", type=int, required=True)
args = parser.parse_args()

H=args.height
W=args.width
MODEL=args.model
model = onnx.load(f'{MODEL}')
model_simp, check = simplify(
    model,
    input_shapes={
        "input.1": [1,3,H,W],
    }
)
basename_without_ext = os.path.splitext(os.path.basename(MODEL))[0].split('_')[0]
onnx.save(model_simp, f'{basename_without_ext}_{H}x{W}.onnx')