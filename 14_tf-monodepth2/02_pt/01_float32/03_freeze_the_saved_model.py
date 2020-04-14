### tf-nightly==2.2.0.dev20200410

import tensorflow as tf
import os
import shutil
from tensorflow.python import ops

def get_graph_def_from_file(graph_filepath):
  tf.compat.v1.reset_default_graph()
  with ops.Graph().as_default():
    with tf.io.gfile.GFile(graph_filepath, 'rb') as f:
      graph_def = tf.compat.v1.GraphDef()
      graph_def.ParseFromString(f.read())
      return graph_def

def convert_graph_def_to_saved_model(export_dir, graph_filepath, input_name, outputs):
  graph_def = get_graph_def_from_file(graph_filepath)
  with tf.compat.v1.Session(graph=tf.Graph()) as session:
    tf.import_graph_def(graph_def, name='')
    tf.compat.v1.saved_model.simple_save(
        session,
        export_dir,# change input_image to node.name if you know the name
        inputs={input_name: session.graph.get_tensor_by_name('{}:0'.format(node.name))
            for node in graph_def.node if node.op=='Placeholder'},
        outputs={t.rstrip(":0"):session.graph.get_tensor_by_name(t) for t in outputs}
     )
    print('Optimized graph converted to SavedModel!')

# Look up the name of the placeholder for the input node
graph_def=get_graph_def_from_file('./model.latest/frozen_model.pb')
input_name=""
for node in graph_def.node:
    if node.op=='Placeholder':
        print("##### frozen_model - Input Node Name #####", node.name) # this will be the input node
        input_name=node.name

# frozen_model output names - colormap and depth
output_node_names = ['monodepth2_model/strided_slice','monodepth2_model/ResizeBilinear']
outputs = ['monodepth2_model/strided_slice:0','monodepth2_model/ResizeBilinear:0']
# convert this to a TF Serving compatible mode - frozen_model
shutil.rmtree('./saved_model', ignore_errors=True)
convert_graph_def_to_saved_model('./saved_model', './model.latest/frozen_model.pb', input_name, outputs)

# # frozen_model output names - colormap only
# output_node_names = ['monodepth2_model/ResizeBilinear']
# outputs = ['monodepth2_model/ResizeBilinear:0']
# # convert this to a TF Serving compatible mode - frozen_model
# shutil.rmtree('./saved_model', ignore_errors=True)
# convert_graph_def_to_saved_model('./saved_model', './model.latest/frozen_model.pb', input_name, outputs)