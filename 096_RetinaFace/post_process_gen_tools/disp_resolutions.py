import os
import onnx

files = sorted([
    f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f)) and '.onnx' in f
])

for file in files:
    model = onnx.load(file)
    height = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    width = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    boxes = model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
    print(f'"{height} {width} {boxes}"')
