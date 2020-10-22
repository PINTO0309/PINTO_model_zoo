### tensorflow==2.3.1

import tensorflow.compat.v1 as tf
import shutil

def get_graph_def_from_file(graph_filepath):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        with tf.gfile.GFile(graph_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def

def convert_graph_def_to_saved_model(export_dir, graph_filepath, input_name, outputs):
    graph_def = get_graph_def_from_file(graph_filepath)
    with tf.Session(graph=tf.Graph()) as session:
        tf.import_graph_def(graph_def, name='')
        tf.saved_model.simple_save(
            session,
            export_dir,
            inputs={input_name: session.graph.get_tensor_by_name('{}:0'.format(node.name))
                for node in graph_def.node if node.op=='Placeholder'},
            outputs={t.rstrip(":0"):session.graph.get_tensor_by_name(t) for t in outputs}
        )
        print('Optimized graph converted to SavedModel!')

shutil.rmtree('saved_model_480x640_fullint', ignore_errors=True)
convert_graph_def_to_saved_model('saved_model', 'model_float32.pb', 'inputs', ['Identity:0'])
