import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

with tf.compat.v1.Session() as sess:

    # shape=[1, 300, 300, 3] -> shape=[1, 256, 256, 3]
    # name='normalized_input_image_tensor' specifies the placeholder name of the converted model
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[1, 256, 256, 3], name='normalized_input_image_tensor')
    with tf.io.gfile.GFile('./tflite_graph_300.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

    # 'normalized_input_image_tensor:0' specifies the placeholder name of the model before conversion
    tf.graph_util.import_graph_def(graph_def, input_map={'normalized_input_image_tensor:0': inputs}, name='')
    print([n for n in tf.compat.v1.get_default_graph().as_graph_def().node if n.name == 'normalized_input_image_tensor'])

    # Delete Placeholder "normalized_input_image_tensor" before conversion
    # see: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms
    # TransformGraph(
    #     graph_def(),
    #     input_op_name,
    #     output_op_names,
    #     conversion options
    # )
    optimized_graph_def = TransformGraph(
                              tf.compat.v1.get_default_graph().as_graph_def(),
                              'normalized_input_image_tensor',
                              ['raw_outputs/box_encodings','raw_outputs/class_predictions'],
                              ['strip_unused_nodes(type=float, shape="1,256,256,3")'])

    tf.io.write_graph(optimized_graph_def, './', 'tflite_graph_256.pb', as_text=False)
