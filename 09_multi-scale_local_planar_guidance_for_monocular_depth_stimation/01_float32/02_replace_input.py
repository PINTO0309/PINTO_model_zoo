import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

with tf.compat.v1.Session() as sess:

    # shape=[1, ?, ?, 3] -> shape=[1, 480, 640, 3]
    # name='image' specifies the placeholder name of the converted model
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[1, 480, 640, 3], name='input')
    with tf.io.gfile.GFile('./models/graph.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

    # 'image:0' specifies the placeholder name of the model before conversion
    tf.graph_util.import_graph_def(graph_def, input_map={'IteratorGetNext:0': inputs}, name='')
    print([n for n in tf.compat.v1.get_default_graph().as_graph_def().node if n.name == 'IteratorGetNext'])

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
                              ['model/decoder/mul_16','model/decoder/truediv_3','model/decoder/truediv_9','model/decoder/truediv_15'],
                              ['strip_unused_nodes(type=float, shape="1,480,640,3")'])

    tf.io.write_graph(optimized_graph_def, './models', 'bts_densenet161_480_640.pbtxt', as_text=True)