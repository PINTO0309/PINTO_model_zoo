import onnx
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('file_path')
args = parser.parse_args()

model = onnx.load(args.file_path)
model = onnx.shape_inference.infer_shapes(model)
onnx.save(model, args.file_path)