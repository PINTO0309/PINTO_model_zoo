### tensorflow-gpu==1.15.2

import tensorflow as tf
from model.head.yolov3 import YOLOV3

pb_file = "weights/yolov3_lite_freeze_graph.pb"
ckpt_file = "weights/yolo.ckpt-60-0.7911"
output_node_names = ["input/input_data", "YoloV3/pred_sbbox/concat_2", "YoloV3/pred_mbbox/concat_2", "YoloV3/pred_lbbox/concat_2"]

with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, shape=(1, 416, 416, 3), name='input_data')
    training = tf.constant(False, dtype=tf.bool, name='training')
_, _, _, pred_sbbox, pred_mbbox, pred_lbbox = YOLOV3(training).build_nework(input_data)


sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
tf.global_variables_initializer()
tf.local_variables_initializer()
saver.restore(sess, ckpt_file)

graphdef = sess.graph.as_graph_def()
for node in graphdef.node:
    if node.op == 'RefSwitch':
        node.op = 'Switch'
        for index in range(len(node.input)):
            if 'moving_' in node.input[index]:
                node.input[index] = node.input[index] + '/read'
    elif node.op == 'AssignSub':
        node.op = 'Sub'
        if 'use_locking' in node.attr: del node.attr['use_locking']
    elif node.op == 'AssignAdd':
        node.op = 'Add'
        if 'use_locking' in node.attr: del node.attr['use_locking']
    elif node.op == 'Assign':
        node.op = 'Identity'
        if 'use_locking' in node.attr: del node.attr['use_locking']
        if 'validate_shape' in node.attr: del node.attr['validate_shape']
        if len(node.input) == 2:
            # input0: ref: Should be from a Variable node. May be uninitialized.
            # input1: value: The value to be assigned to the variable.
            node.input[0] = node.input[1]
            del node.input[1]

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = graphdef,
                            output_node_names = output_node_names)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())
