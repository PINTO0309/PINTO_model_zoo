import tensorflow as tf
import os

graph_def = tf.compat.v1.GraphDef()


# Import the TF graph
with tf.io.gfile.GFile('pose_detection_128x128_float32.pb', 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
