### tensorflow==2.3.0

import tensorflow as tf
import shutil

def get_graph_def_from_file(graph_filepath):
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Graph().as_default():
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
            export_dir,
            inputs={input_name: session.graph.get_tensor_by_name('{}:0'.format(node.name))
                for node in graph_def.node if node.op=='Placeholder'},
            outputs={t.rstrip(":0"):session.graph.get_tensor_by_name(t) for t in outputs}
        )
        print('Optimized graph converted to SavedModel!')

shutil.rmtree('saved_model_512x1024', ignore_errors=True)
convert_graph_def_to_saved_model('saved_model_512x1024', 'bisenetv2_cityscapes_frozen_512x1024.pb', 'input_tensor', ['final_output:0'])

shutil.rmtree('saved_model_480x640', ignore_errors=True)
convert_graph_def_to_saved_model('saved_model_480x640', 'bisenetv2_cityscapes_frozen_480x640.pb', 'input_tensor', ['final_output:0'])

shutil.rmtree('saved_model_256x256', ignore_errors=True)
convert_graph_def_to_saved_model('saved_model_256x256', 'bisenetv2_cityscapes_frozen_256x256.pb', 'input_tensor', ['final_output:0'])