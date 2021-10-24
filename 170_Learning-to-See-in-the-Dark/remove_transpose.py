import onnx_graphsurgeon as gs
import onnx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--onnx_file_path", required=True, type=str)
parser.add_argument("--remove_node_name", required=True, type=str)
args = parser.parse_args()

graph = gs.import_onnx(onnx.load(args.onnx_file_path))

for i in graph.nodes:
    print(i.name)

remove_node = [
    node for node in graph.nodes if node.name == args.remove_node_name
][0]

# Get the input node of the fake node
# Node provides i() and o() functions that can optionally
# be provided an index (default is 0)
# These serve as convenience functions for the alternative,
# which would be to fetch the input/output
# tensor first, then fetch the input/output node of the tensor.
# For example, node.i() is equivalent to node.inputs[0].inputs[0]
inp_node = remove_node.i()

# Reconnect the input node to the output tensors of the fake node,
# so that the first identity node in the example graph now
# skips over the fake node.
inp_node.outputs = remove_node.outputs
remove_node.outputs.clear()

# Remove the fake node from the graph completely
graph.cleanup()

h = graph.inputs[0].shape[2]
w = graph.inputs[0].shape[3]
graph.outputs[0].shape = [1,3,h*2,w*2]
print(graph.outputs)

onnx.save(gs.export_onnx(graph), args.onnx_file_path)
