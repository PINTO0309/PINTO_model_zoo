import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

with tf.compat.v1.Session() as sess:

    # shape=[1, ?, ?, 3] -> shape=[1, 224, 224, 3]
    # name='image' specifies the placeholder name of the converted model
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[1, 224, 224, 3], name='input')
    with tf.io.gfile.GFile('./mobilenet_v2_0.5_224_frozen.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

    # 'image:0' specifies the placeholder name of the model before conversion
    tf.graph_util.import_graph_def(graph_def, input_map={'input:0': inputs}, name='')
    print([n for n in tf.compat.v1.get_default_graph().as_graph_def().node if n.name == 'input'])

    # Delete Placeholder "IteratorGetNext" before conversion
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
                              ['MobilenetV2/Predictions/Reshape_1'],
                              ['strip_unused_nodes(type=float, shape="1,224,224,3")'])

    tf.io.write_graph(optimized_graph_def, './', 'mobilenet_v2_0.5_224_optimization.pb', as_text=False)