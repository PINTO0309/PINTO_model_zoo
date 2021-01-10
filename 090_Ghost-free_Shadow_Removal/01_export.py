vgg19_path = 'imagenet-vgg-verydeep-19.mat'
pretrain_model_path = 'srdplus-pretrained/'
sample_path = 'ghost-free-shadow-removal/Samples'

import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from networks import build_aggasatt_joint

tf.disable_eager_execution()

with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    input=tf.placeholder(tf.float32,shape=[1,256,256,3])
    shadow_free_image=build_aggasatt_joint(input,64,vgg19_path)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
idtd_ckpt=tf.train.get_checkpoint_state(pretrain_model_path)
saver_restore=tf.train.Saver([var for var in tf.trainable_variables()])
print('loaded '+idtd_ckpt.model_checkpoint_path)
saver_restore.restore(sess,idtd_ckpt.model_checkpoint_path)

frozen_graph_def = tf.graph_util.convert_variables_to_constants(
  sess,
  sess.graph_def,
  ['g_conv_img/BiasAdd', 'g_conv_mask/BiasAdd'])

# Save the frozen graph
with open('shadow_removal.pb', 'wb') as f:
  f.write(frozen_graph_def.SerializeToString())
