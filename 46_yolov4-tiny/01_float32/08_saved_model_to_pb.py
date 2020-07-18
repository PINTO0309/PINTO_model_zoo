### tensorflow-gpu==1.15.2

import sys
import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

with tf.compat.v1.Session() as sess:
    with gfile.FastGFile('saved_model/saved_model.pb', 'rb') as f:
        data = compat.as_bytes(f.read())
        sm = saved_model_pb2.SavedModel()
        sm.ParseFromString(data)
        if 1 != len(sm.meta_graphs):
            print('More than one graph found. Not sure which to write')
            sys.exit(1)

    # 'image:0' specifies the placeholder name of the model before conversion
    tf.graph_util.import_graph_def(sm.meta_graphs[0].graph_def, name='')

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
                              'serving_default_input_1',
                              ['StatefulPartitionedCall:0','StatefulPartitionedCall:1'],
                              ['strip_unused_nodes(type=float, shape="1,416,416,3")'])

    tf.io.write_graph(optimized_graph_def, './', 'yolov4_tiny_voc_416x416.pb', as_text=False)