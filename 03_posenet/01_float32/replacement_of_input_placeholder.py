import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

with tf.compat.v1.Session() as sess:

    # shape=[1, ?, ?, 3] -> shape=[1, 513, 513, 3]
    # name='image' specifies the placeholder name of the converted model
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[1, 513, 513, 3], name='image')
    with tf.io.gfile.GFile('./model-mobilenet_v1_101.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

    # 'image:0' specifies the placeholder name of the model before conversion
    tf.graph_util.import_graph_def(graph_def, input_map={'image:0': inputs}, name='')
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
                              ['heatmap','offset_2','displacement_fwd_2','displacement_bwd_2'],
                              ['strip_unused_nodes(type=float, shape="1,513,513,3")'])

    tf.io.write_graph(optimized_graph_def, './', 'model-mobilenet_v1_101_513.pb', as_text=False)