### tensorflow==2.2.0

import tensorflow as tf
from tensorflow.python import ops
import shutil

def get_graph_def_from_file(graph_filepath):
    tf.compat.v1.reset_default_graph()
    with ops.Graph().as_default():
        with tf.compat.v1.gfile.GFile(graph_filepath, 'rb') as f:
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

shutil.rmtree('saved_model_mosaic', ignore_errors=True)
convert_graph_def_to_saved_model('saved_model_mosaic', 'mosaic.pb', 'input1', ['output1:0'])

"""
$ saved_model_cli show --dir saved_model_mosaic --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input1'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 3, 224, 224)
        name: input1:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['output1'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 3, 224, 224)
        name: output1:0
  Method name is: tensorflow/serving/predict
"""