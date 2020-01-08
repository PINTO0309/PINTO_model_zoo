import tensorflow as tf
import os
import shutil
#from tensorflow.python.saved_model import tag_constants
#from tensorflow.python.tools import freeze_graph
from tensorflow.python import ops

def get_graph_def_from_file(graph_filepath):
  tf.reset_default_graph()
  with ops.Graph().as_default():
    with tf.gfile.GFile(graph_filepath, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      return graph_def

tf.compat.v1.enable_eager_execution()

# Look up the name of the placeholder for the input node
graph_def=get_graph_def_from_file('./models/graph.pb')
input_name=""
for node in graph_def.node:
    if node.op=='Placeholder':
        print("##### Input Node Name #####", node.name) # this will be the input node
        input_name=node.name

# +++++ INPUT ++++++
#inputs_name=IteratorGetNext:0
#inputs_shape= (?, ?, ?, 3)

# +++++ OUTPUT +++++
# depth= model/decoder/mul_16:0
# pred_8x8= model/decoder/truediv_3:0
# pred_4x4= model/decoder/truediv_9:0
# pred_2x2= model/decoder/truediv_15:0
