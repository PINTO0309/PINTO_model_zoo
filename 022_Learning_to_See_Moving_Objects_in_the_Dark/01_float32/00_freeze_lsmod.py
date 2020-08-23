import tensorflow as tf
from network import network
import numpy as np
from config import *

#######################################################################################
### $ python3 freeze_lsmod.py
#######################################################################################

def main():

    graph = tf.Graph()
    with graph.as_default():

        #in_image = tf.compat.v1.placeholder(tf.float32, [None, TEST_CROP_FRAME, None, None, 4], name='input')
        #gt_image = tf.compat.v1.placeholder(tf.float32, [None, TEST_CROP_FRAME, None, None, 3], name='gt')
        in_image = tf.compat.v1.placeholder(tf.float32, [None, CROP_FRAME, 256, 256, 4], name='input')
        gt_image = tf.compat.v1.placeholder(tf.float32, [None, CROP_FRAME, 256, 256, 3], name='gt')

        out_image = network(in_image)

        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        sess  = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        saver.restore(sess, './1_checkpoint/16_bit_HE_to_HE_gt/model.ckpt')
        saver.save(sess, './1_checkpoint/16_bit_HE_to_HE_gt/modelfilnal.ckpt')

        graphdef = tf.compat.v1.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['output'])
        tf.io.write_graph(graphdef, './1_checkpoint/16_bit_HE_to_HE_gt', 'lsmod_256.pb', as_text=False)

if __name__ == '__main__':
    main()
