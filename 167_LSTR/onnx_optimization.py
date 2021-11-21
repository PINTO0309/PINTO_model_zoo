import onnx_graphsurgeon as gs
import onnx
import argparse
import os
import base64
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--onnx_file_path", required=True, type=str)
args = parser.parse_args()

remove_node_names = ['Split_988', 'Split_990']
indices_change_initializer_names = ['1498'] # 'Gather_975', 'Gather_977'

# Remove Split
graph = gs.import_onnx(onnx.load(args.onnx_file_path))
remove_nodes = [
    node for node in graph.nodes if node.name in remove_node_names
]
for remove_node in remove_nodes:
    inp_node = remove_node.i()
    inp_node.outputs = remove_node.outputs
    remove_node.outputs.clear()
graph.cleanup()
onnx.save(gs.export_onnx(graph), args.onnx_file_path)

# Change indices Gather -1 to 1
import json
from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse
onnx_model = onnx.load(args.onnx_file_path)
s = MessageToJson(onnx_model)
onnx_json = json.loads(s)
indices = 1
for i in onnx_json['graph']['initializer']:
    if i['name'] in indices_change_initializer_names:
        i['rawData'] = base64.b64encode(np.array(indices).tobytes()).decode('utf-8')
onnx_model = Parse(json.dumps(onnx_json), onnx.ModelProto())
onnx.save(onnx_model, args.onnx_file_path)
print('Optimaization done')