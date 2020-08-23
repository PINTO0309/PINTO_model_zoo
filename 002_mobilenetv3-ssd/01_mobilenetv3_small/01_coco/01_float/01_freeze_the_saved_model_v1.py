import tensorflow as tf
import os
import shutil
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph
from tensorflow.python import ops
from tensorflow.tools.graph_transforms import TransformGraph


def get_graph_def_from_file(graph_filepath):
  tf.reset_default_graph()
  with ops.Graph().as_default():
    with tf.gfile.GFile(graph_filepath, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      return graph_def


def convert_graph_def_to_saved_model(export_dir, graph_filepath, input_name, outputs):
  graph_def = get_graph_def_from_file(graph_filepath)
  with tf.Session(graph=tf.Graph()) as session:
    tf.import_graph_def(graph_def, name='')
    tf.compat.v1.saved_model.simple_save(
        session,
        export_dir,# change input_image to node.name if you know the name
        inputs={input_name: session.graph.get_tensor_by_name('{}:0'.format(node.name))
            for node in graph_def.node if node.op=='Placeholder'},
        outputs={t.rstrip(":0"):session.graph.get_tensor_by_name(t) for t in outputs}
    )
    print('Optimized graph converted to SavedModel!')

tf.compat.v1.enable_eager_execution()

# Look up the name of the placeholder for the input node
graph_def=get_graph_def_from_file('./tflite_graph.pb')
input_name=""
for node in graph_def.node:
    if node.op=='Placeholder':
        print("##### frozen_inference_graph - Input Node Name #####", node.name) # this will be the input node
        input_name=node.name

# mobilenetv2_fsd2018_41cls output names
outputs = ['raw_outputs/box_encodings:0','raw_outputs/class_predictions:0']

# convert this to a TF Serving compatible mode - mobilenetv2_fsd2018_41cls
shutil.rmtree('./saved_model', ignore_errors=True)
convert_graph_def_to_saved_model('./saved_model', './tflite_graph.pb', input_name, outputs)
