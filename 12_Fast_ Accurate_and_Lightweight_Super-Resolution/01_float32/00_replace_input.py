import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

with tf.compat.v1.Session() as sess:

    # shape=[?, ?, ?, 1] -> shape=[1, 160, 240, 1]
    # shape=[?, ?, ?, 1] -> shape=[1, 320, 480, 1]
    # name='image' specifies the placeholder name of the converted model
    inputs1 = tf.compat.v1.placeholder(tf.float32, shape=[1, 160, 240, 1], name='input_image_evaluate_y')
    inputs2 = tf.compat.v1.placeholder(tf.float32, shape=[1, 320, 480, 2], name='input_image_evaluate_pbpr')
    with tf.io.gfile.GFile('./FALSR-A.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

    # 'image:0' specifies the placeholder name of the model before conversion
    tf.graph_util.import_graph_def(graph_def, input_map={'input_image_evaluate_y:0': inputs1, 'input_image_evaluate_pbpr:0': inputs2}, name='')
    print([n for n in tf.compat.v1.get_default_graph().as_graph_def().node if n.name == 'input_image_evaluate_y'])
    print([n for n in tf.compat.v1.get_default_graph().as_graph_def().node if n.name == 'input_image_evaluate_pbpr'])

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
                              ['input_image_evaluate_y','input_image_evaluate_pbpr'],
                              ['test_sr_evaluator_i1_b0_g/target'],
                              ['strip_unused_nodes(type=float, shape="1,160,240,1")'])

    tf.io.write_graph(optimized_graph_def, './', 'FALSR-A_optimization.pb', as_text=False)


