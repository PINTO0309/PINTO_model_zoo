### tensorflow-gpu==1.15.2

import tensorflow as tf
import model
import nets
import util

pb_file_kitti_depth = "struct2depth_128x416_kitti_depth.pb"
pb_file_kitti_egomotion = "struct2depth_128x416_kitti_egomotion.pb"
model_ckpt_kitti = "struct2depth_model_kitti/model-199160"

pb_file_cityscapes_depth = "struct2depth_128x416_cityscapes_depth.pb"
pb_file_cityscapes_egomotion = "struct2depth_128x416_cityscapes_egomotion.pb"
model_ckpt_cityscapes = "struct2depth_model_cityscapes/model-154688"

### depth_input_node_name = depth_prediction/raw_input:0
### egomotion_input_node_name = truediv_1:0

output_node_names_depth = ["truediv"]
output_node_names_egomotion = ["egomotion_prediction/pose_exp_net/pose/concat"]

graph = tf.Graph()
with graph.as_default():
    inference_model = model.Model(is_training=False,
                                batch_size=1,
                                img_height=128,
                                img_width=416,
                                seq_length=3,
                                architecture=nets.RESNET,
                                imagenet_norm=True,
                                use_skip=True,
                                joint_encoder=False)
    vars_to_restore = util.get_vars_to_save_and_restore(model_ckpt_kitti)
    saver = tf.train.Saver(vars_to_restore)
    with tf.Session() as sess:
        saver.restore(sess, model_ckpt_kitti)
        converted_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
                                    input_graph_def  = graph.as_graph_def(),
                                    output_node_names = output_node_names_depth)
        with tf.gfile.GFile(pb_file_kitti_depth, "wb") as f:
            f.write(converted_graph_def.SerializeToString())

        converted_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
                                    input_graph_def  = graph.as_graph_def(),
                                    output_node_names = output_node_names_egomotion)
        with tf.gfile.GFile(pb_file_kitti_egomotion, "wb") as f:
            f.write(converted_graph_def.SerializeToString())



graph = tf.Graph()
with graph.as_default():
    inference_model = model.Model(is_training=False,
                                batch_size=1,
                                img_height=128,
                                img_width=416,
                                seq_length=3,
                                architecture=nets.RESNET,
                                imagenet_norm=True,
                                use_skip=True,
                                joint_encoder=False)
    vars_to_restore = util.get_vars_to_save_and_restore(model_ckpt_cityscapes)
    saver = tf.train.Saver(vars_to_restore)
    with tf.Session() as sess:
        saver.restore(sess, model_ckpt_cityscapes)
        converted_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
                                    input_graph_def  = graph.as_graph_def(),
                                    output_node_names = output_node_names_depth)
        with tf.gfile.GFile(pb_file_cityscapes_depth, "wb") as f:
            f.write(converted_graph_def.SerializeToString())

        converted_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
                                    input_graph_def  = graph.as_graph_def(),
                                    output_node_names = output_node_names_egomotion)
        with tf.gfile.GFile(pb_file_cityscapes_egomotion, "wb") as f:
            f.write(converted_graph_def.SerializeToString())
