### tensorflow-gpu==1.15.2

import tensorflow as tf
from wrappers import ModelPipeline

model = ModelPipeline()

#####################################################################################
pb_file_det_model = "freeze_graph_detnet.pb"
ckpt_file_det_model = "model/detnet/detnet.ckpt"
output_node_names_det_model = ["prior_based_hand/strided_slice_6", "prior_based_hand/strided_slice_7"]

converted_graph_def = tf.graph_util.convert_variables_to_constants(model.det_model.sess,
                            input_graph_def  = model.det_model.graph.as_graph_def(),
                            output_node_names = output_node_names_det_model)

with tf.gfile.GFile(pb_file_det_model, "wb") as f:
    f.write(converted_graph_def.SerializeToString())
#####################################################################################
pb_file_ik_model = "reeze_graph_iknet.pb"
ckpt_file_ik_model = "model/iknet/iknet.ckpt"
output_node_names_ik_model = ["network/Select"]

converted_graph_def = tf.graph_util.convert_variables_to_constants(model.ik_model.sess,
                            input_graph_def  = model.ik_model.graph.as_graph_def(),
                            output_node_names = output_node_names_ik_model)

with tf.gfile.GFile(pb_file_ik_model, "wb") as f:
    f.write(converted_graph_def.SerializeToString())