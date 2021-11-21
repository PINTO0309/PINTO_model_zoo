import onnx_graphsurgeon as gs
import onnx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--onnx_file_path", required=True, type=str)
parser.add_argument("--remove_node_name", required=True, type=str)
args = parser.parse_args()

graph = gs.import_onnx(onnx.load(args.onnx_file_path))
remove_node = None
remove_node_idx = -1
for idx, node in enumerate(graph.nodes):
    if node.name == args.remove_node_name:
        remove_node = node
        remove_node_idx = idx
        break
graph.inputs[0].dtype = graph.nodes[remove_node_idx+1].inputs[0].dtype
graph.nodes[remove_node_idx+1].inputs[0] = graph.inputs[0]
remove_node.outputs.clear()
graph.cleanup()
onnx.save(gs.export_onnx(graph), args.onnx_file_path)