### tensorflow==2.3.1

### https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html
### https://google.github.io/mediapipe/solutions/pose

### https://www.tensorflow.org/api_docs/python/tf/keras/Model
### https://www.tensorflow.org/lite/guide/ops_compatibility

### https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/Add
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/Concatenate
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer

### How to initialize a convolution layer with an arbitrary kernel in Keras? https://stackoverrun.com/ja/q/12269118

###  saved_model_cli show --dir saved_model_hand_landmark_new/ --tag_set serve --signature_def serving_default

import tensorflow as tf
import tf_slim as slim
import tensorflow_datasets as tfds
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, ReLU, MaxPool2D, Reshape, Concatenate, AveragePooling2D, Dense, Lambda, Conv2DTranspose
from tensorflow.keras.initializers import Constant
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import sys


height = 128
width  = 128
convert_mode = 'normal' # normal or edgetpu

# False: No optimization to EdgeTPU, True: Optimizing for EdgeTPU.
if convert_mode == 'normal':
    optimizing_hardswish_for_edgetpu_flg = False
elif convert_mode == 'edgetpu':
    optimizing_hardswish_for_edgetpu_flg = True
else:
    optimizing_hardswish_for_edgetpu_flg = False


def optimizing_hardswish_for_edgetp(input_op):
    ret_op = None
    if not optimizing_hardswish_for_edgetpu_flg:
        ret_op = input_op * tf.nn.relu6(input_op + 3) * 0.16666667
    else:
        ret_op = input_op * tf.nn.relu6(input_op + 3) * 0.16666666
    return ret_op

def upsampling2d_bilinear(x, upsampling_factor_height, upsampling_factor_width):
    h = x.shape[1] * upsampling_factor_height
    w = x.shape[2] * upsampling_factor_width
    return tf.compat.v1.image.resize_bilinear(x, (h, w))

def upsampling2d_nearest(x, upsampling_factor_height, upsampling_factor_width):
    h = x.shape[1] * upsampling_factor_height
    w = x.shape[2] * upsampling_factor_width
    return tf.compat.v1.image.resize_nearest_neighbor(x, (h, w))


# Input
inputs = Input(shape=(height, width, 3), batch_size=1, name='input_1')

# Block_01 - input_1 - h_swish
conv_1_1 = Conv2D(filters=16,
                 kernel_size=[3, 3],
                 dilation_rate=[1, 1],
                 activation=None,
                 padding='same',
                 strides=[2, 2],
                 kernel_initializer=Constant(np.load('weights/conv2d_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_Bias')))(inputs)

hardswish_1_1 = optimizing_hardswish_for_edgetp(conv_1_1)


# Block_02 - h_swish - multiply
conv_2_1 = Conv2D(filters=16,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation='relu',
                 padding='same',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_1_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_1_Bias')))(hardswish_1_1)

depthconv_2_1 = DepthwiseConv2D(kernel_size=[3, 3],
                               dilation_rate=[1, 1],
                               activation='relu',
                               padding='same',
                               strides=[2, 2],
                               depth_multiplier=1,
                               depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_Kernel')),
                               bias_initializer=Constant(np.load('weights/depthwise_conv2d_Bias')))(conv_2_1)

avgpool_2_1 = AveragePooling2D(pool_size=[32, 32],
                               padding='valid',
                               strides=[32, 32])(depthconv_2_1)

dense_2_1 = Dense(units=16,
                  activation='relu',
                  use_bias=True,
                  kernel_initializer=Constant(np.load('weights/dense_Kernel').transpose(1,0)),
                  bias_initializer=Constant(np.load('weights/dense_Bias')))(avgpool_2_1)

dense_2_2 = Dense(units=16,
                  activation=None,
                  use_bias=True,
                  kernel_initializer=Constant(np.load('weights/dense_1_Kernel').transpose(1,0)),
                  bias_initializer=Constant(np.load('weights/dense_1_Bias')))(dense_2_1)

sigm_2_1 = tf.math.sigmoid(dense_2_2)

mul_2_1 = tf.math.multiply(depthconv_2_1, sigm_2_1)

# Block_03 - multiply - conv2d_1
conv_3_1 = Conv2D(filters=16,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation=None,
                 padding='same',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_2_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_2_Bias')))(mul_2_1)

# Block_04 - conv2d_2 - conv2d_4
conv_4_1 = Conv2D(filters=72,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation='relu',
                 padding='same',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_3_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_3_Bias')))(conv_3_1)

depthconv_4_1 = DepthwiseConv2D(kernel_size=[3, 3],
                               dilation_rate=[1, 1],
                               activation='relu',
                               padding='same',
                               strides=[2, 2],
                               depth_multiplier=1,
                               depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_1_Kernel')),
                               bias_initializer=Constant(np.load('weights/depthwise_conv2d_1_Bias')))(conv_4_1)

conv_4_2 = Conv2D(filters=24,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation=None,
                 padding='same',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_4_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_4_Bias')))(depthconv_4_1)

# Block_05 - conv2d_4 - conv2d_6
conv_5_1 = Conv2D(filters=88,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation='relu',
                 padding='same',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_5_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_5_Bias')))(conv_4_2)

depthconv_5_1 = DepthwiseConv2D(kernel_size=[3, 3],
                               dilation_rate=[1, 1],
                               activation='relu',
                               padding='same',
                               strides=[1, 1],
                               depth_multiplier=1,
                               depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_2_Kernel')),
                               bias_initializer=Constant(np.load('weights/depthwise_conv2d_2_Bias')))(conv_5_1)

conv_5_2 = Conv2D(filters=24,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation=None,
                 padding='same',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_6_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_6_Bias')))(depthconv_5_1)

# Block_06 - conv2d_4 conv2d_6 - add__xeno_compat__1
add_6_1 = Add()([conv_4_2, conv_5_2])

# Block_07 - add__xeno_compat__1 - multiply_1
conv_7_1 = Conv2D(filters=96,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation=None,
                 padding='same',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_7_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_7_Bias')))(add_6_1)

hardswish_7_1 = optimizing_hardswish_for_edgetp(conv_7_1)

depthconv_7_1 = DepthwiseConv2D(kernel_size=[5, 5],
                               dilation_rate=[1, 1],
                               activation=None,
                               padding='same',
                               strides=[2, 2],
                               depth_multiplier=1,
                               depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_3_Kernel')),
                               bias_initializer=Constant(np.load('weights/depthwise_conv2d_3_Bias')))(hardswish_7_1)

hardswish_7_2 = optimizing_hardswish_for_edgetp(depthconv_7_1)

avgpool_7_1 = AveragePooling2D(pool_size=[8, 8],
                               padding='valid',
                               strides=[8, 8])(hardswish_7_2)

dense_7_1 = Dense(units=96,
                  activation='relu',
                  use_bias=True,
                  kernel_initializer=Constant(np.load('weights/dense_2_Kernel').transpose(1,0)),
                  bias_initializer=Constant(np.load('weights/dense_2_Bias')))(avgpool_7_1)

dense_7_2 = Dense(units=96,
                  activation=None,
                  use_bias=True,
                  kernel_initializer=Constant(np.load('weights/dense_3_Kernel').transpose(1,0)),
                  bias_initializer=Constant(np.load('weights/dense_3_Bias')))(dense_7_1)

sigm_7_1 = tf.math.sigmoid(dense_7_2)

mul_7_1 = tf.math.multiply(hardswish_7_2, sigm_7_1)

# Block_08 - multiply_1 - conv2d_8
conv_8_1 = Conv2D(filters=32,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation=None,
                 padding='same',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_8_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_8_Bias')))(mul_7_1)

# Block_09 - conv2d_8 - multiply_2
conv_9_1 = Conv2D(filters=128,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation=None,
                 padding='same',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_9_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_9_Bias')))(conv_8_1)

hardswish_9_1 = optimizing_hardswish_for_edgetp(conv_9_1)

depthconv_9_1 = DepthwiseConv2D(kernel_size=[5, 5],
                               dilation_rate=[1, 1],
                               activation=None,
                               padding='same',
                               strides=[1, 1],
                               depth_multiplier=1,
                               depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_4_Kernel')),
                               bias_initializer=Constant(np.load('weights/depthwise_conv2d_4_Bias')))(hardswish_9_1)

hardswish_9_2 = optimizing_hardswish_for_edgetp(depthconv_9_1)

avgpool_9_1 = AveragePooling2D(pool_size=[8, 8],
                               padding='valid',
                               strides=[8, 8])(hardswish_9_2)

dense_9_1 = Dense(units=128,
                  activation='relu',
                  use_bias=True,
                  kernel_initializer=Constant(np.load('weights/dense_4_Kernel').transpose(1,0)),
                  bias_initializer=Constant(np.load('weights/dense_4_Bias')))(avgpool_9_1)

dense_9_2 = Dense(units=128,
                  activation=None,
                  use_bias=True,
                  kernel_initializer=Constant(np.load('weights/dense_5_Kernel').transpose(1,0)),
                  bias_initializer=Constant(np.load('weights/dense_5_Bias')))(dense_9_1)

sigm_9_1 = tf.math.sigmoid(dense_9_2)

mul_9_1 = tf.math.multiply(hardswish_9_2, sigm_9_1)

# Block_10 - multiply_2 - add_1__xeno_compat__1
conv_10_1 = Conv2D(filters=32,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation=None,
                 padding='same',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_10_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_10_Bias')))(mul_9_1)

add_10_1 = Add()([conv_8_1, conv_10_1])

# Block_11 - add_1__xeno_compat__1 - multiply_3
conv_11_1 = Conv2D(filters=128,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation=None,
                 padding='same',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_11_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_11_Bias')))(add_10_1)

hardswish_11_1 = optimizing_hardswish_for_edgetp(conv_11_1)

depthconv_11_1 = DepthwiseConv2D(kernel_size=[5, 5],
                               dilation_rate=[1, 1],
                               activation=None,
                               padding='same',
                               strides=[1, 1],
                               depth_multiplier=1,
                               depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_5_Kernel')),
                               bias_initializer=Constant(np.load('weights/depthwise_conv2d_5_Bias')))(hardswish_11_1)

hardswish_11_2 = optimizing_hardswish_for_edgetp(depthconv_11_1)

avgpool_11_1 = AveragePooling2D(pool_size=[8, 8],
                               padding='valid',
                               strides=[8, 8])(hardswish_11_2)

dense_11_1 = Dense(units=128,
                  activation='relu',
                  use_bias=True,
                  kernel_initializer=Constant(np.load('weights/dense_6_Kernel').transpose(1,0)),
                  bias_initializer=Constant(np.load('weights/dense_6_Bias')))(avgpool_11_1)

dense_11_2 = Dense(units=128,
                  activation=None,
                  use_bias=True,
                  kernel_initializer=Constant(np.load('weights/dense_7_Kernel').transpose(1,0)),
                  bias_initializer=Constant(np.load('weights/dense_7_Bias')))(dense_11_1)

sigm_11_1 = tf.math.sigmoid(dense_11_2)

mul_11_1 = tf.math.multiply(hardswish_11_2, sigm_11_1)

# Block_12 - multiply_3 - add_2__xeno_compat__1
conv_12_1 = Conv2D(filters=32,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation=None,
                 padding='same',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_12_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_12_Bias')))(mul_11_1)

add_12_1 = Add()([add_10_1, conv_12_1])

# Block_13 - add_2__xeno_compat__1 - multiply_4
conv_13_1 = Conv2D(filters=96,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation=None,
                 padding='same',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_13_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_13_Bias')))(add_12_1)

hardswish_13_1 = optimizing_hardswish_for_edgetp(conv_13_1)

depthconv_13_1 = DepthwiseConv2D(kernel_size=[5, 5],
                               dilation_rate=[1, 1],
                               activation=None,
                               padding='same',
                               strides=[1, 1],
                               depth_multiplier=1,
                               depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_6_Kernel')),
                               bias_initializer=Constant(np.load('weights/depthwise_conv2d_6_Bias')))(hardswish_13_1)

hardswish_13_2 = optimizing_hardswish_for_edgetp(depthconv_13_1)

avgpool_13_1 = AveragePooling2D(pool_size=[8, 8],
                               padding='valid',
                               strides=[8, 8])(hardswish_13_2)

dense_13_1 = Dense(units=96,
                  activation='relu',
                  use_bias=True,
                  kernel_initializer=Constant(np.load('weights/dense_8_Kernel').transpose(1,0)),
                  bias_initializer=Constant(np.load('weights/dense_8_Bias')))(avgpool_13_1)

dense_13_2 = Dense(units=96,
                  activation=None,
                  use_bias=True,
                  kernel_initializer=Constant(np.load('weights/dense_9_Kernel').transpose(1,0)),
                  bias_initializer=Constant(np.load('weights/dense_9_Bias')))(dense_13_1)

sigm_13_1 = tf.math.sigmoid(dense_13_2)

mul_13_1 = tf.math.multiply(hardswish_13_2, sigm_13_1)

# Block_14 - multiply_4 - add_3__xeno_compat__1
conv_14_1 = Conv2D(filters=32,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation=None,
                 padding='same',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_14_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_14_Bias')))(mul_13_1)

add_14_1 = Add()([add_12_1, conv_14_1])

# Block_15 - add_3__xeno_compat__1 - multiply_5
conv_15_1 = Conv2D(filters=72,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation=None,
                 padding='same',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_15_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_15_Bias')))(add_14_1)

hardswish_15_1 = optimizing_hardswish_for_edgetp(conv_15_1)

depthconv_15_1 = DepthwiseConv2D(kernel_size=[5, 5],
                               dilation_rate=[1, 1],
                               activation=None,
                               padding='same',
                               strides=[1, 1],
                               depth_multiplier=1,
                               depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_7_Kernel')),
                               bias_initializer=Constant(np.load('weights/depthwise_conv2d_7_Bias')))(hardswish_15_1)

hardswish_15_2 = optimizing_hardswish_for_edgetp(depthconv_15_1)

avgpool_15_1 = AveragePooling2D(pool_size=[8, 8],
                               padding='valid',
                               strides=[8, 8])(hardswish_15_2)

dense_15_1 = Dense(units=72,
                  activation='relu',
                  use_bias=True,
                  kernel_initializer=Constant(np.load('weights/dense_10_Kernel').transpose(1,0)),
                  bias_initializer=Constant(np.load('weights/dense_10_Bias')))(avgpool_15_1)

dense_15_2 = Dense(units=72,
                  activation=None,
                  use_bias=True,
                  kernel_initializer=Constant(np.load('weights/dense_11_Kernel').transpose(1,0)),
                  bias_initializer=Constant(np.load('weights/dense_11_Bias')))(dense_15_1)

sigm_15_1 = tf.math.sigmoid(dense_15_2)

mul_15_1 = tf.math.multiply(hardswish_15_2, sigm_15_1)

# Block_16 - multiply_5 - multiply_6
conv_16_1 = Conv2D(filters=24,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation=None,
                 padding='same',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_16_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_16_Bias')))(mul_15_1)

#===============================
conv_16_2 = Conv2D(filters=128,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation='relu',
                 padding='same',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_17_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_17_Bias')))(conv_16_1)

#===============================
avgpool_16_1 = AveragePooling2D(pool_size=[5, 5],
                               padding='valid',
                               strides=[3, 3])(conv_16_1)

conv_16_3 = Conv2D(filters=128,
                 kernel_size=[1, 1],
                 dilation_rate=[1, 1],
                 activation='relu',
                 padding='valid',
                 strides=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_18_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_18_Bias')))(avgpool_16_1)

sigm_16_1 = tf.math.sigmoid(conv_16_3)

resize_16_1 = Lambda(upsampling2d_bilinear, arguments={'upsampling_factor_height': 4, 'upsampling_factor_width':  4})(sigm_16_1)

mul_16_1 = tf.math.multiply(conv_16_2, resize_16_1)

# Block_17 - multiply_5 - conv2d_19
resize_17_1 = Lambda(upsampling2d_bilinear, arguments={'upsampling_factor_height': 2, 'upsampling_factor_width':  2})(mul_16_1)

conv_17_1 = Conv2D(filters=24,
                   kernel_size=[1, 1],
                   dilation_rate=[1, 1],
                   activation=None,
                   padding='valid',
                   strides=[1, 1],
                   kernel_initializer=Constant(np.load('weights/conv2d_19_Kernel').transpose(1,2,3,0)),
                   bias_initializer=Constant(np.load('weights/conv2d_19_Bias')))(resize_17_1)

# Block_18 - add__xeno_compat__1 - multiply_7
concat_18_1 = tf.concat([add_6_1, conv_17_1], axis=-1)

avgpool_18_1 = AveragePooling2D(pool_size=[16, 16],
                               padding='valid',
                               strides=[16, 16])(concat_18_1)

conv_18_1 = Conv2D(filters=24,
                   kernel_size=[1, 1],
                   dilation_rate=[1, 1],
                   activation='relu',
                   padding='valid',
                   strides=[1, 1],
                   kernel_initializer=Constant(np.load('weights/conv2d_20_Kernel').transpose(1,2,3,0)),
                   bias_initializer=Constant(np.load('weights/conv2d_20_Bias')))(avgpool_18_1)

conv_18_2 = Conv2D(filters=24,
                   kernel_size=[1, 1],
                   dilation_rate=[1, 1],
                   activation=None,
                   padding='valid',
                   strides=[1, 1],
                   kernel_initializer=Constant(np.load('weights/conv2d_21_Kernel').transpose(1,2,3,0)),
                   bias_initializer=Constant(np.load('weights/conv2d_21_Bias')))(conv_18_1)

sigm_18_1 = tf.math.sigmoid(conv_18_2)

mul_18_1 = tf.math.multiply(add_6_1, sigm_18_1)

# Block_19 - multiply_7 - add_4__xeno_compat__1
add_19_1 = Add()([conv_17_1, mul_18_1])

# Block_20 - add_4__xeno_compat__1 - add_5__xeno_compat__1
conv_20_1 = Conv2D(filters=24,
                   kernel_size=[1, 1],
                   dilation_rate=[1, 1],
                   activation=None,
                   padding='valid',
                   strides=[1, 1],
                   kernel_initializer=Constant(np.load('weights/conv2d_22_Kernel').transpose(1,2,3,0)),
                   bias_initializer=Constant(np.load('weights/conv2d_22_Bias')))(add_19_1)

depthconv_20_1 = DepthwiseConv2D(kernel_size=[3, 3],
                               dilation_rate=[1, 1],
                               activation='relu',
                               padding='same',
                               strides=[1, 1],
                               depth_multiplier=1,
                               depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_8_Kernel')),
                               bias_initializer=Constant(np.load('weights/depthwise_conv2d_8_Bias')))(conv_20_1)

add_20_1 = Add()([conv_20_1, depthconv_20_1])

# Block_21 - add_5__xeno_compat__1 - conv2d_23
resize_21_1 = Lambda(upsampling2d_bilinear, arguments={'upsampling_factor_height': 2, 'upsampling_factor_width':  2})(add_20_1)

conv_21_1 = Conv2D(filters=16,
                   kernel_size=[1, 1],
                   dilation_rate=[1, 1],
                   activation=None,
                   padding='valid',
                   strides=[1, 1],
                   kernel_initializer=Constant(np.load('weights/conv2d_23_Kernel').transpose(1,2,3,0)),
                   bias_initializer=Constant(np.load('weights/conv2d_23_Bias')))(resize_21_1)

# Block_22 - add__xeno_compat__1 - multiply_8
concat_22_1 = tf.concat([conv_3_1, conv_21_1], axis=-1)

avgpool_22_1 = AveragePooling2D(pool_size=[32, 32],
                               padding='valid',
                               strides=[32, 32])(concat_22_1)

conv_22_1 = Conv2D(filters=16,
                   kernel_size=[1, 1],
                   dilation_rate=[1, 1],
                   activation='relu',
                   padding='valid',
                   strides=[1, 1],
                   kernel_initializer=Constant(np.load('weights/conv2d_24_Kernel').transpose(1,2,3,0)),
                   bias_initializer=Constant(np.load('weights/conv2d_24_Bias')))(avgpool_22_1)

conv_22_2 = Conv2D(filters=16,
                   kernel_size=[1, 1],
                   dilation_rate=[1, 1],
                   activation=None,
                   padding='valid',
                   strides=[1, 1],
                   kernel_initializer=Constant(np.load('weights/conv2d_25_Kernel').transpose(1,2,3,0)),
                   bias_initializer=Constant(np.load('weights/conv2d_25_Bias')))(conv_22_1)

sigm_22_1 = tf.math.sigmoid(conv_22_2)

mul_22_1 = tf.math.multiply(conv_3_1, sigm_22_1)

# Block_23 - multiply_8 - add_6__xeno_compat__1
add_23_1 = Add()([conv_21_1, mul_22_1])

# Block_24 - add_6__xeno_compat__1 - add_7__xeno_compat__1
conv_24_1 = Conv2D(filters=16,
                   kernel_size=[1, 1],
                   dilation_rate=[1, 1],
                   activation=None,
                   padding='valid',
                   strides=[1, 1],
                   kernel_initializer=Constant(np.load('weights/conv2d_26_Kernel').transpose(1,2,3,0)),
                   bias_initializer=Constant(np.load('weights/conv2d_26_Bias')))(add_23_1)

depthconv_24_1 = DepthwiseConv2D(kernel_size=[3, 3],
                               dilation_rate=[1, 1],
                               activation='relu',
                               padding='same',
                               strides=[1, 1],
                               depth_multiplier=1,
                               depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_9_Kernel')),
                               bias_initializer=Constant(np.load('weights/depthwise_conv2d_9_Bias')))(conv_24_1)

add_24_1 = Add()([conv_24_1, depthconv_24_1])

# Block_25 - add_7__xeno_compat__1 - conv2d_27
resize_25_1 = Lambda(upsampling2d_bilinear, arguments={'upsampling_factor_height': 2, 'upsampling_factor_width':  2})(add_24_1)

conv_25_1 = Conv2D(filters=16,
                   kernel_size=[1, 1],
                   dilation_rate=[1, 1],
                   activation=None,
                   padding='valid',
                   strides=[1, 1],
                   kernel_initializer=Constant(np.load('weights/conv2d_27_Kernel').transpose(1,2,3,0)),
                   bias_initializer=Constant(np.load('weights/conv2d_27_Bias')))(resize_25_1)

# Block_26 - h_swish conv2d_27 - multiply_9
concat_26_1 = tf.concat([hardswish_1_1, conv_25_1], axis=-1)

avgpool_26_1 = AveragePooling2D(pool_size=[64, 64],
                               padding='valid',
                               strides=[64, 64])(concat_26_1)

conv_26_1 = Conv2D(filters=16,
                   kernel_size=[1, 1],
                   dilation_rate=[1, 1],
                   activation='relu',
                   padding='valid',
                   strides=[1, 1],
                   kernel_initializer=Constant(np.load('weights/conv2d_28_Kernel').transpose(1,2,3,0)),
                   bias_initializer=Constant(np.load('weights/conv2d_28_Bias')))(avgpool_26_1)

conv_26_2 = Conv2D(filters=16,
                   kernel_size=[1, 1],
                   dilation_rate=[1, 1],
                   activation=None,
                   padding='valid',
                   strides=[1, 1],
                   kernel_initializer=Constant(np.load('weights/conv2d_29_Kernel').transpose(1,2,3,0)),
                   bias_initializer=Constant(np.load('weights/conv2d_29_Bias')))(conv_26_1)

sigm_26_1 = tf.math.sigmoid(conv_26_2)

mul_26_1 = tf.math.multiply(hardswish_1_1, sigm_26_1)

# Block_27 - multiply_9 - add_8__xeno_compat__1
add_27_1 = Add()([conv_25_1, mul_26_1])

# Block_28 - add_6__xeno_compat__1 - add_7__xeno_compat__1
conv_28_1 = Conv2D(filters=16,
                   kernel_size=[1, 1],
                   dilation_rate=[1, 1],
                   activation=None,
                   padding='valid',
                   strides=[1, 1],
                   kernel_initializer=Constant(np.load('weights/conv2d_30_Kernel').transpose(1,2,3,0)),
                   bias_initializer=Constant(np.load('weights/conv2d_30_Bias')))(add_27_1)

depthconv_28_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                 dilation_rate=[1, 1],
                                 activation='relu',
                                 padding='same',
                                 strides=[1, 1],
                                 depth_multiplier=1,
                                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_10_Kernel')),
                                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_10_Bias')))(conv_28_1)

add_28_1 = Add()([conv_28_1, depthconv_28_1])

# Block_29 - add_9__xeno_compat__1  - segment
convtrans_29_1 = tf.nn.conv2d_transpose(input=add_28_1,
                                        filters=np.load('weights/segment_Kernel').transpose(1,2,0,3).astype(np.float32),
                                        output_shape=[1, height, width, 2],
                                        strides=[2, 2],
                                        padding='SAME',
                                        dilations=[1, 1])
# add_29_1 = Add(name='BaiasAdd')([convtrans_29_1, np.load('weights/segment_Bias').reshape(1,1,1,2).astype(np.float32)])
# add_29_1 = Add()([convtrans_29_1, np.load('weights/segment_Bias')[::-1].reshape(1,1,1,2).astype(np.float32)])
# add_29_1 = Add(name='BaiasAdd')([id_29_1, np.load('weights/segment_Bias').astype(np.float32)])
add_29_1 = tf.math.add(convtrans_29_1, np.load('weights/segment_Bias').astype(np.float32))

model = Model(inputs=inputs, outputs=[add_29_1])
model.summary()




saved_model_path = f'saved_model_{height}x{width}'

tf.saved_model.save(model, saved_model_path)
model.save(f'{saved_model_path}/segm_lite_v509_{height}x{width}_float32.h5')

full_model = tf.function(lambda inputs: model(inputs))
full_model = full_model.get_concrete_function(inputs = (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)))
frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
frozen_func.graph.as_graph_def()
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir='.',
                    name=f'{saved_model_path}/segm_lite_v509_{height}x{width}_float32.pb',
                    as_text=False)

# No Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
with open(f'{saved_model_path}/segm_lite_v509_{height}x{width}_float32.tflite', 'wb') as w:
    w.write(tflite_model)
print(f'tflite convert complete! - {saved_model_path}/segm_lite_v509_{height}x{width}_float32.tflite')


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
with open(f'{saved_model_path}/segm_lite_v509_{height}x{width}_weight_quant.tflite', 'wb') as w:
    w.write(tflite_model)
print(f'Weight Quantization complete! - {saved_model_path}/segm_lite_v509_{height}x{width}_weight_quant.tflite')

# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_quant_model = converter.convert()
with open(f'{saved_model_path}/segm_lite_v509_{height}x{width}_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print(f'Float16 Quantization complete! - {saved_model_path}/segm_lite_v509_{height}x{width}_float16_quant.tflite')

def representative_dataset_gen():
  for data in raw_test_data.take(10):
    image = data['image'].numpy()
    image = tf.image.resize(image, (height, width))
    image = image[np.newaxis,:,:,:]
    image = image / 255
    yield [image]

raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="validation", data_dir="~/TFDS", download=True)


# Integer Quantization - Input/Output=float32 - tf-nightly
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open(f'{saved_model_path}/segm_lite_v509_{height}x{width}_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print(f'Integer Quantization complete! - {saved_model_path}/segm_lite_v509_{height}x{width}_integer_quant.tflite')

# Full Integer Quantization - Input/Output=int8 - tf-nightly
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open(f'{saved_model_path}/segm_lite_v509_{height}x{width}_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print(f'Full Integer Quantization complete! - {saved_model_path}/segm_lite_v509_{height}x{width}_full_integer_quant.tflite')


# EdgeTPU - tf-nightly
import subprocess
result = subprocess.check_output(["edgetpu_compiler", "-s", f"{saved_model_path}/segm_lite_v509_{height}x{width}_full_integer_quant.tflite"])
print(result.decode('utf-8'))





# TensorFlow.js convert
import subprocess
try:
    result = subprocess.check_output(['tensorflowjs_converter',
                                    '--input_format', 'tf_saved_model',
                                    '--output_format', 'tfjs_graph_model',
                                    '--signature_name', 'serving_default',
                                    '--saved_model_tags', 'serve',
                                    saved_model_path, f'{saved_model_path}/tfjs_model_float32'],
                                    stderr=subprocess.PIPE).decode('utf-8')
    print(result)
    print(f'TensorFlow.js convertion complete! - {saved_model_path}/tfjs_model_float32')
except subprocess.CalledProcessError as e:
    print(f'ERROR:', e.stderr.decode('utf-8'))
    import traceback
    traceback.print_exc()
try:
    result = subprocess.check_output(['tensorflowjs_converter',
                                    '--quantize_float16',
                                    '--input_format', 'tf_saved_model',
                                    '--output_format', 'tfjs_graph_model',
                                    '--signature_name', 'serving_default',
                                    '--saved_model_tags', 'serve',
                                    saved_model_path, f'{saved_model_path}/tfjs_model_float16'],
                                    stderr=subprocess.PIPE).decode('utf-8')
    print(result)
    print(f'TensorFlow.js convertion complete! - {saved_model_path}/tfjs_model_float16')
except subprocess.CalledProcessError as e:
    print(f'ERROR:', e.stderr.decode('utf-8'))
    import traceback
    traceback.print_exc()

# TF-TRT (TensorRT) convert
try:
    def input_fn():
        input_shapes = []
        for tf_input in model.inputs:
            input_shapes.append(np.zeros(tf_input.shape).astype(np.float32))
        yield input_shapes

    params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP32', maximum_cached_engines=10000)
    converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=saved_model_path, conversion_params=params)
    converter.convert()
    converter.build(input_fn=input_fn)
    converter.save(f'{saved_model_path}/tensorrt_saved_model_float32')
    print(f'TF-TRT (TensorRT) convertion complete! - {saved_model_path}/tensorrt_saved_model_float32')
    params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP16', maximum_cached_engines=10000)
    converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=saved_model_path, conversion_params=params)
    converter.convert()
    converter.build(input_fn=input_fn)
    converter.save(f'{saved_model_path}/tensorrt_saved_model_float16')
    print(f'TF-TRT (TensorRT) convertion complete! - {saved_model_path}/tensorrt_saved_model_float16')
except Exception as e:
    print(f'ERROR:', e)
    import traceback
    traceback.print_exc()
    print(f'The binary versions of TensorFlow and TensorRT may not be compatible. Please check the version compatibility of each package.')

# CoreML convert
try:
    import coremltools as ct 
    mlmodel = ct.convert(saved_model_path, source='tensorflow')
    mlmodel.save(f'{saved_model_path}/model_coreml_float32.mlmodel')
    print(f'CoreML convertion complete! - {saved_model_path}/model_coreml_float32.mlmodel')
except Exception as e:
    print(f'ERROR:', e)
    import traceback
    traceback.print_exc()
