import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

with tf.compat.v1.Session() as sess:

    # shape=[1, ?, ?, 3] -> shape=[1, 128, 128, 3]
    # name='image' specifies the placeholder name of the converted model
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[1, 128, 128, 3], name='input_image_tensor')
    with tf.io.gfile.GFile('frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

    # 'image:0' specifies the placeholder name of the model before conversion
    tf.graph_util.import_graph_def(graph_def, input_map={'input_to_float:0': inputs}, name='')
    print([n for n in tf.compat.v1.get_default_graph().as_graph_def().node if n.name == 'input_image_tensor'])

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
                              'input_image_tensor',
                              ['logits/BiasAdd'],
                              ['strip_unused_nodes(type=float, shape="1,128,128,3")'])

    tf.io.write_graph(optimized_graph_def, './', 'head_pose_estimator.pb', as_text=False)