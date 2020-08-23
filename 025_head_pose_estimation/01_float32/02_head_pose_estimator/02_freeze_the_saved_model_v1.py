### tensorflow-gpu==1.15.2

import tensorflow as tf
import os
import shutil
from tensorflow.python import ops

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
    print('Graph converted to SavedModel!')

tf.compat.v1.enable_eager_execution()

input_name="input_image_tensor"
outputs = ['logits/BiasAdd:0']
shutil.rmtree('./saved_model', ignore_errors=True)
convert_graph_def_to_saved_model('./saved_model', './head_pose_estimator.pb', input_name, outputs)

"""
$ saved_model_cli show --dir saved_model --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_image_tensor'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 128, 128, 3)
        name: input_image_tensor:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['logits/BiasAdd'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 136)
        name: logits/BiasAdd:0
  Method name is: tensorflow/serving/predict
"""

