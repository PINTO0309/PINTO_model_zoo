import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
import numpy as np

GRAPH_PB_PATH = './mobilenet_v3_small.pb' #path to your .pb file

with tf.Session() as sess:
  with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes=[n for n in graph_def.node]
    wts = [n for n in graph_nodes if n.op=='Const']
    for n in wts:
        if n.name == 'anchors':
            print("Name of the node - %s" % n.name)
            print("Value - ")
            anchors = tensor_util.MakeNdarray(n.attr['value'].tensor)
            print("anchors.shape =", anchors.shape)
            print(anchors)
            np.save('./anchors.npy', anchors)
            np.savetxt('./anchors.csv', anchors, delimiter=',')
            break