### tensorflow-gpu==1.15.2

import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

with tf.compat.v1.Session() as sess:

    # shape=[?, ?, ?, 3] -> shape=[1, 256, 256, 3]
    # name='image' specifies the placeholder name of the converted model
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[1, 256, 256, 3], name='tower_0/images')
    with tf.io.gfile.GFile('detector.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

    # 'image:0' specifies the placeholder name of the model before conversion
    tf.graph_util.import_graph_def(graph_def, input_map={'tower_0/images:0': inputs}, name='')
    print([n for n in tf.compat.v1.get_default_graph().as_graph_def().node if n.name == 'tower_0/images'])

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
                              'tower_0/images',
                              ['tower_0/boxes','tower_0/labels','tower_0/scores','tower_0/num_detections'],
                              ['strip_unused_nodes(type=float, shape="1,256,256,3")'])

    tf.io.write_graph(optimized_graph_def, './', 'detector_256x256.pb', as_text=False)