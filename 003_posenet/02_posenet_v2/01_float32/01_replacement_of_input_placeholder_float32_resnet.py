### tensorflow-gpu==1.15.2

import sys
import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

#tf.compat.v1.disable_eager_execution()

with tf.compat.v1.Session() as sess:

    # shape=[1, ?, ?, 3] -> shape=[1, 513, 513, 3]
    # name='image' specifies the placeholder name of the converted model
    
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[1, 513, 513, 3], name='image')
    #inputs = tf.compat.v1.placeholder(tf.float32, shape=[1, 385, 385, 3], name='image')
    #inputs = tf.compat.v1.placeholder(tf.float32, shape=[1, 321, 321, 3], name='image')
    #inputs = tf.compat.v1.placeholder(tf.float32, shape=[1, 257, 257, 3], name='image')
    #inputs = tf.compat.v1.placeholder(tf.float32, shape=[1, 225, 225, 3], name='image')

    with gfile.FastGFile('_tf_models/posenet/resnet50_float/stride32/saved_model.pb', 'rb') as f:
        data = compat.as_bytes(f.read())
        sm = saved_model_pb2.SavedModel()
        sm.ParseFromString(data)
        if 1 != len(sm.meta_graphs):
            print('More than one graph found. Not sure which to write')
            sys.exit(1)

    # 'sub_2:0' specifies the placeholder name of the model before conversion
    tf.graph_util.import_graph_def(sm.meta_graphs[0].graph_def, input_map={'sub_2:0': inputs}, name='')
    print([n for n in tf.compat.v1.get_default_graph().as_graph_def().node if n.name == 'image'])

    # Delete Placeholder "image" before conversion
    # see: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms
    # TransformGraph(
    #     graph_def(),
    #     input_name,
    #     output_names,
    #     conversion options
    # )
    optimized_graph_def = TransformGraph(
                              tf.compat.v1.get_default_graph().as_graph_def(),
                              'image',
                              ['float_heatmaps','float_short_offsets','resnet_v1_50/displacement_fwd_2/BiasAdd','resnet_v1_50/displacement_bwd_2/BiasAdd'],
                              ['strip_unused_nodes(type=float, shape="1,513,513,3")'])

    tf.io.write_graph(optimized_graph_def, './', 'posenet_resnet50_32_513.pb', as_text=False)
