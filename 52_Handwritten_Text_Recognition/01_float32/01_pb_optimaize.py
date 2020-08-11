### tensorflow-gpu==1.15.2

import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

with tf.compat.v1.Session() as sess:

    # shape=[1, 128, 32, 1]
    with tf.io.gfile.GFile('simpleHTR_freeze_graph.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

    tf.graph_util.import_graph_def(graph_def)#, input_map={'image:0': inputs}, name='')

    # Delete Placeholder "image" before conversion
    # see: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms
    # TransformGraph(
    #     graph_def(),
    #     input_op_name,
    #     output_op_names,
    #     conversion options
    # )
    optimized_graph_def = TransformGraph(
                              tf.compat.v1.get_default_graph().as_graph_def(),
                              'input',
                              ['CTCGreedyDecoder','CTCGreedyDecoder:1','CTCGreedyDecoder:2','CTCGreedyDecoder:3'],
                              ['strip_unused_nodes(type=float, shape="1,128,32,1")'])

    tf.io.write_graph(optimized_graph_def, '.', 'simpleHTR_freeze_graph_opt.pb', as_text=False)