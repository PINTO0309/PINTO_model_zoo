import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.python.framework import graph_util

graph = tf.get_default_graph()
sess = tf.Session()
saver = tf.train.import_meta_graph('saved_model_dronet/DroNet_car.ckpt.meta')
saver.restore(sess, 'saved_model_dronet/DroNet_car.ckpt')

graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['dronet/convolutional9/BiasAdd'])
tf.train.write_graph(graph_def, 'export', 'DroNet_car.pb', as_text=False)

graph = tf.get_default_graph()
sess = tf.Session()
saver = tf.train.import_meta_graph('saved_model/DroNetV3_car.ckpt.meta')
saver.restore(sess, 'saved_model/DroNetV3_car.ckpt')

graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['dronetv3/convolutional12/BiasAdd'])
tf.train.write_graph(graph_def, 'export', 'DroNetV3_car.pb', as_text=False)