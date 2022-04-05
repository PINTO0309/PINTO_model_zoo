import onnx
import onnx_graphsurgeon as gs
import numpy as np
from collections import OrderedDict
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('filepath')
parser.add_argument('classes')
args = parser.parse_args()

graph = gs.import_onnx(onnx.load(args.filepath))
pre_node_names = ['Softmax_101']
pre_nodes = [node for node in graph.nodes if node.name in pre_node_names]
dummy_reducemean_out = gs.Variable(
    name="10000",
    dtype=np.float32,
    shape=[1,args.classes]
)
dummy_reducemean = gs.Node(
    op="ReduceMean",
    name="head_reducemean",
    attrs=OrderedDict(
        [
            ('axes', 0),
            ('keepdims', 1),
        ]
    ),
    inputs=pre_nodes[0].outputs,
    outputs=[dummy_reducemean_out]
)
graph.nodes.append(dummy_reducemean)
graph.outputs[0] = dummy_reducemean.outputs[0]
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), f'{args.filepath}')
