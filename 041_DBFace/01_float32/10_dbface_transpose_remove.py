### tensorflow==2.2.0

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
import copy

INPUT_GRAPH_DEF_FILE = "dbface.pb"
OUTPUT_GRAPH_DEF_FILE = "dbface_transpose_remove.pb"

with tf.compat.v1.Session() as sess: #@@@@@@@@@@@@@@@@@@

    graph = tf.compat.v1.GraphDef()

    with gfile.FastGFile(INPUT_GRAPH_DEF_FILE, 'rb') as f:
        data = compat.as_bytes(f.read())
        graph.ParseFromString(data)

    inputs = tf.compat.v1.placeholder(tf.float32, shape=[1, 3, 480, 640], name='x')
    new_graph = graph_pb2.GraphDef()

    for node in graph.node:
        if node.name in ['model/823/transpose','model/823/transpose_1',
                        'model/841/transpose','model/841/transpose_1',
                        'model/859/transpose','model/859/transpose_1']:
            pass

        elif node.name == 'model/823/resize/ResizeNearestNeighbor':
            node.input[0] = 'model/822/truediv'
            new_graph.node.extend([copy.deepcopy(node)])
        elif node.name == 'model/841/resize/ResizeNearestNeighbor':
            node.input[0] = 'model/840/add'
            new_graph.node.extend([copy.deepcopy(node)])
        elif node.name == 'model/859/resize/ResizeNearestNeighbor':
            node.input[0] = 'model/858/add'
            new_graph.node.extend([copy.deepcopy(node)])

        elif node.name == 'model/824_pad/Pad':
            node.input[0] = 'model/823/resize/ResizeNearestNeighbor'
            new_graph.node.extend([copy.deepcopy(node)])
        elif node.name == 'model/842_pad/Pad':
            node.input[0] = 'model/841/resize/ResizeNearestNeighbor'
            new_graph.node.extend([copy.deepcopy(node)])
        elif node.name == 'model/860_pad/Pad':
            node.input[0] = 'model/859/resize/ResizeNearestNeighbor'
            new_graph.node.extend([copy.deepcopy(node)])
        else:
            new_graph.node.extend([copy.deepcopy(node)])

    # for node in graph.node:
    #     if node.name in ['model/831/truediv','model/840/add','model/841/resize/ResizeNearestNeighbor']:
    #         print("node:", node)


    tf.graph_util.import_graph_def(new_graph, input_map={'x:0': inputs}, name='')

    with gfile.GFile(OUTPUT_GRAPH_DEF_FILE, "wb") as f:
        f.write(new_graph.SerializeToString())
