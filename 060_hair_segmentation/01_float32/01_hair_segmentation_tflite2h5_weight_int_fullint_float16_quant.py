### tensorflow==2.3.0

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

### https://github.com/google/mediapipe/issues/245
### https://github.com/mvoelk/keras_layers

### How to initialize a convolution layer with an arbitrary kernel in Keras? https://stackoverrun.com/ja/q/12269118

###  saved_model_cli show --dir saved_model/ --tag_set serve --signature_def serving_default

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, Add, ReLU, PReLU, MaxPool2D, Reshape, Concatenate, Layer
from tensorflow.keras.initializers import Constant
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops
import numpy as np
import sys
import cv2

# tmp = np.load('weights/depthwise_conv2d_Kernel')
# print(tmp.shape)
# print(tmp)

# def init_f(shape, dtype=None):
#        ker = np.load('weights/depthwise_conv2d_Kernel')
#        print(shape)
#        return ker

# sys.exit(0)

# class MaxPoolingWithArgmax2D(Layer):
#     def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
#         super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
#         self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
#         self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
#         self.padding = conv_utils.normalize_padding(padding)

#     def call(self, inputs, **kwargs):
#         ksize = [1, self.pool_size[0], self.pool_size[1], 1]
#         strides = [1, self.strides[0], self.strides[1], 1]
#         padding = self.padding.upper()
#         output, argmax = nn_ops.max_pool_with_argmax(inputs, ksize, strides, padding)
#         # output, argmax = tf.raw_ops.MaxPoolWithArgmax(inputs, ksize, strides, padding)
#         argmax = tf.cast(argmax, K.floatx())
#         return [output, argmax]
    
#     def compute_output_shape(self, input_shape):
#         ratio = (1, 2, 2, 1)
#         output_shape = [dim // ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
#         output_shape = tuple(output_shape)
#         return [output_shape, output_shape]

#     def compute_mask(self, inputs, mask=None):
#         return 2 * [None]
    
#     def get_config(self):
#         config = super(MaxPoolingWithArgmax2D, self).get_config()
#         config.update({
#             'pool_size': self.pool_size,
#             'strides': self.strides,
#             'padding': self.padding,
#         })
#         return config


def max_pooling_with_argmax2d(input):
    net_main = tf.nn.max_pool(input,
                        ksize=[1,2,2,1],
                        strides=[1,2,2,1],
                        padding='SAME')

    input_shape = input.get_shape().as_list()
    mask_shape = [input_shape[0], input_shape [1]//2,input_shape[2]//2, input_shape[3]]
    pooling_indices = tf.zeros(mask_shape, dtype=tf.int64)
    for n in range(mask_shape[0]):
        for i in range(mask_shape[1]):
            for j in range(mask_shape[2]):
                in_indices = [ [n, w, h] for w in range(i*2, i*2+2) for h in range(j*2, j*2+2)]
                slice = tf.gather_nd(input, in_indices)
                argmax = tf.argmax(slice, axis=0)
                indices_location = [[n, i, j, d] for d in range(input_shape[3])]
                sparse_indices = tf.SparseTensor(indices=indices_location, values=argmax, dense_shape=mask_shape)
                pooling_indices = tf.compat.v1.sparse_add(pooling_indices, sparse_indices)
    return [net_main, pooling_indices]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        
        mask = tf.cast(mask, 'int32')
        input_shape = tf.shape(updates, out_type='int32')
        #  calculation new shape
        if output_shape is None:
            output_shape = (input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3])
        
        # calculation indices for batch, height, width and feature maps
        one_like_mask = K.ones_like(mask, dtype='int32')
        batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = K.reshape(tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype='int32')
        f = one_like_mask * feature_range
        
        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
        values = K.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret
    
    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        output_shape = [mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]]
        return tuple(output_shape)
    
    def get_config(self):
        config = super(MaxUnpooling2D, self).get_config()
        config.update({
            'size': self.size,
        })
        return config


height = 512
width  = 512
inputs = Input(shape=(height, width, 4), batch_size=1, name='input')

# Block_01
conv1_1 = Conv2D(filters=8, kernel_size=[2, 2], strides=[2, 2], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_Bias')))(inputs)
prelu1_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_Alpha')), shared_axes=[1, 2])(conv1_1)

conv1_2 = Conv2D(filters=32, kernel_size=[2, 2], strides=[2, 2], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_1_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_1_Bias')))(prelu1_1)
prelu1_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_1_Alpha')), shared_axes=[1, 2])(conv1_2)


# Block_02
conv2_1 = Conv2D(filters=16, kernel_size=[2, 2], strides=[2, 2], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_2_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_2_Bias')))(prelu1_2)
prelu2_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_2_Alpha')), shared_axes=[1, 2])(conv2_1)
depthconv2_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_Bias')))(prelu2_1)

conv2_2 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_3_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_3_Bias')))(depthconv2_1)
prelu2_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_3_Alpha')), shared_axes=[1, 2])(conv2_2)
depthconv2_2 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_1_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_1_Bias')))(prelu2_2)

prelu2_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_4_Alpha')), shared_axes=[1, 2])(depthconv2_2)
conv2_3 = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_4_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_4_Bias')))(prelu2_3)

maxpoolarg2_1 = tf.raw_ops.MaxPoolWithArgmax(input=prelu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# maxpoolarg2_1 = max_pooling_with_argmax2d(prelu1_2)


conv2_4 = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_5_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_5_Bias')))(maxpoolarg2_1[0])

add2_1 = Add()([conv2_3, conv2_4])
prelu2_4 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_5_Alpha')), shared_axes=[1, 2])(add2_1)


# Block_03
conv3_1 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_6_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_6_Bias')))(prelu2_4)
prelu3_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_6_Alpha')), shared_axes=[1, 2])(conv3_1)
depthconv3_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_2_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_2_Bias')))(prelu3_1)

conv3_2 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_7_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_7_Bias')))(depthconv3_1)
prelu3_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_7_Alpha')), shared_axes=[1, 2])(conv3_2)
depthconv3_2 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_3_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_3_Bias')))(prelu3_2)

prelu3_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_8_Alpha')), shared_axes=[1, 2])(depthconv3_2)
conv3_3 = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_8_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_8_Bias')))(prelu3_3)

add3_1 = Add()([conv3_3, prelu2_4])
prelu3_4 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_9_Alpha')), shared_axes=[1, 2])(add3_1)


# Block_04
conv4_1 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_9_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_9_Bias')))(prelu3_4)
prelu4_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_10_Alpha')), shared_axes=[1, 2])(conv4_1)
depthconv4_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_4_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_4_Bias')))(prelu4_1)

conv4_2 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_10_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_10_Bias')))(depthconv4_1)
prelu4_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_11_Alpha')), shared_axes=[1, 2])(conv4_2)
depthconv4_2 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_5_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_5_Bias')))(prelu4_2)

prelu4_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_12_Alpha')), shared_axes=[1, 2])(depthconv4_2)
conv4_3 = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_11_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_11_Bias')))(prelu4_3)

add4_1 = Add()([conv4_3, prelu3_4])
prelu4_4 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_13_Alpha')), shared_axes=[1, 2])(add4_1)


# Block_05
conv5_1 = Conv2D(filters=32, kernel_size=[2, 2], strides=[2, 2], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_12_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_12_Bias')))(prelu4_4)
prelu5_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_14_Alpha')), shared_axes=[1, 2])(conv5_1)
depthconv5_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_6_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_6_Bias')))(prelu5_1)

conv5_2 = Conv2D(filters=32, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_13_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_13_Bias')))(depthconv5_1)
prelu5_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_15_Alpha')), shared_axes=[1, 2])(conv5_2)
depthconv5_2 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_7_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_7_Bias')))(prelu5_2)

prelu5_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_16_Alpha')), shared_axes=[1, 2])(depthconv5_2)
conv5_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_14_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_14_Bias')))(prelu5_3)

maxpoolarg5_1 = tf.raw_ops.MaxPoolWithArgmax(input=prelu4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# maxpoolarg5_1 = max_pooling_with_argmax2d(prelu4_4)

conv5_4 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_15_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_15_Bias')))(maxpoolarg5_1[0])

add5_1 = Add()([conv5_3, conv5_4])
prelu5_4 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_17_Alpha')), shared_axes=[1, 2])(add5_1)


# Block_06
conv6_1 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_16_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_16_Bias')))(prelu5_4)
prelu6_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_18_Alpha')), shared_axes=[1, 2])(conv6_1)
depthconv6_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_8_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_8_Bias')))(prelu6_1)

conv6_2 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_17_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_17_Bias')))(depthconv6_1)
prelu6_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_19_Alpha')), shared_axes=[1, 2])(conv6_2)
depthconv6_2 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_9_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_9_Bias')))(prelu6_2)

prelu6_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_20_Alpha')), shared_axes=[1, 2])(depthconv6_2)
conv6_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_18_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_18_Bias')))(prelu6_3)

add6_1 = Add()([conv6_3, prelu5_4])
prelu6_4 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_21_Alpha')), shared_axes=[1, 2])(add6_1)


# Block_07
conv7_1 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_19_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_19_Bias')))(prelu6_4)
prelu7_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_22_Alpha')), shared_axes=[1, 2])(conv7_1)
depthconv7_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_10_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_10_Bias')))(prelu7_1)

conv7_2 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_20_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_20_Bias')))(depthconv7_1)
prelu7_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_23_Alpha')), shared_axes=[1, 2])(conv7_2)
depthconv7_2 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_11_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_11_Bias')))(prelu7_2)

prelu7_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_24_Alpha')), shared_axes=[1, 2])(depthconv7_2)
conv7_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_21_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_21_Bias')))(prelu7_3)

add7_1 = Add()([conv7_3, prelu6_4])
prelu7_4 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_25_Alpha')), shared_axes=[1, 2])(add7_1)


# Block_08
conv8_1 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_22_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_22_Bias')))(prelu7_4)
prelu8_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_26_Alpha')), shared_axes=[1, 2])(conv8_1)
depthconv8_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_12_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_12_Bias')))(prelu8_1)

conv8_2 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_23_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_23_Bias')))(depthconv8_1)
prelu8_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_27_Alpha')), shared_axes=[1, 2])(conv8_2)
depthconv8_2 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_13_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_13_Bias')))(prelu8_2)

prelu8_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_28_Alpha')), shared_axes=[1, 2])(depthconv8_2)
conv8_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_24_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_24_Bias')))(prelu8_3)

add8_1 = Add()([conv8_3, prelu7_4])
prelu8_4 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_29_Alpha')), shared_axes=[1, 2])(add8_1)


# Block_09
conv9_1 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_25_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_25_Bias')))(prelu8_4)
prelu9_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_30_Alpha')), shared_axes=[1, 2])(conv9_1)
depthconv9_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_14_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_14_Bias')))(prelu9_1)

conv9_2 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_26_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_26_Bias')))(depthconv9_1)
prelu9_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_31_Alpha')), shared_axes=[1, 2])(conv9_2)
depthconv9_2 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_15_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_15_Bias')))(prelu9_2)

prelu9_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_32_Alpha')), shared_axes=[1, 2])(depthconv9_2)
conv9_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_27_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_27_Bias')))(prelu9_3)

add9_1 = Add()([conv9_3, prelu8_4])
prelu9_4 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_33_Alpha')), shared_axes=[1, 2])(add9_1)


# Block_10
conv10_1 = Conv2D(filters=16, kernel_size=[2, 2], strides=[2, 2], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_28_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_28_Bias')))(prelu9_4)
prelu10_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_34_Alpha')), shared_axes=[1, 2])(conv10_1)
depthconv10_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_16_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_16_Bias')))(prelu10_1)

conv10_2 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_29_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_29_Bias')))(depthconv10_1)
prelu10_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_35_Alpha')), shared_axes=[1, 2])(conv10_2)
depthconv10_2 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_17_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_17_Bias')))(prelu10_2)

prelu10_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_36_Alpha')), shared_axes=[1, 2])(depthconv10_2)
conv10_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_30_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_30_Bias')))(prelu10_3)

maxpoolarg10_1 = tf.raw_ops.MaxPoolWithArgmax(input=prelu9_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# maxpoolarg10_1 = max_pooling_with_argmax2d(prelu9_4)

add10_1 = Add()([conv10_3, maxpoolarg10_1[0]])
prelu10_4 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_37_Alpha')), shared_axes=[1, 2])(add10_1)


# Block_11
conv11_1 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_31_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_31_Bias')))(prelu10_4)
prelu11_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_38_Alpha')), shared_axes=[1, 2])(conv11_1)
depthconv11_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_18_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_18_Bias')))(prelu11_1)

conv11_2 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_32_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_32_Bias')))(depthconv11_1)
prelu11_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_39_Alpha')), shared_axes=[1, 2])(conv11_2)
depthconv11_2 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_19_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_19_Bias')))(prelu11_2)

prelu11_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_40_Alpha')), shared_axes=[1, 2])(depthconv11_2)
conv11_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_33_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_33_Bias')))(prelu11_3)

add11_1 = Add()([conv11_3, prelu10_4])
prelu11_4 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_41_Alpha')), shared_axes=[1, 2])(add11_1)


# Block_12
conv12_1 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_34_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_34_Bias')))(prelu11_4)
prelu12_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_42_Alpha')), shared_axes=[1, 2])(conv12_1)

conv12_2 = Conv2D(filters=8, kernel_size=[3, 3], strides=[1, 1], padding='same', dilation_rate=[2, 2],
                 kernel_initializer=Constant(np.load('weights/conv2d_35_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_35_Bias')))(prelu12_1)
prelu12_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_43_Alpha')), shared_axes=[1, 2])(conv12_2)

conv12_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_36_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_36_Bias')))(prelu12_2)

add12_1 = Add()([conv12_3, prelu11_4])
prelu12_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_44_Alpha')), shared_axes=[1, 2])(add12_1)


# Block_13
conv13_1 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_37_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_37_Bias')))(prelu12_3)
prelu13_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_45_Alpha')), shared_axes=[1, 2])(conv13_1)
depthconv13_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_20_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_20_Bias')))(prelu13_1)

conv13_2 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_38_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_38_Bias')))(depthconv13_1)
prelu13_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_46_Alpha')), shared_axes=[1, 2])(conv13_2)
conv13_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_39_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_39_Bias')))(prelu13_2)

add13_1 = Add()([conv13_3, prelu12_3])
prelu13_4 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_47_Alpha')), shared_axes=[1, 2])(add13_1)


# Block_14
conv14_1 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_40_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_40_Bias')))(prelu13_4)
prelu14_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_48_Alpha')), shared_axes=[1, 2])(conv14_1)

conv14_2 = Conv2D(filters=8, kernel_size=[3, 3], strides=[1, 1], padding='same', dilation_rate=[4, 4],
                 kernel_initializer=Constant(np.load('weights/conv2d_41_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_41_Bias')))(prelu14_1)
prelu14_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_49_Alpha')), shared_axes=[1, 2])(conv14_2)

conv14_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_42_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_42_Bias')))(prelu14_2)

add14_1 = Add()([conv14_3, prelu13_4])
prelu14_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_50_Alpha')), shared_axes=[1, 2])(add14_1)


# Block_15
conv15_1 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_43_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_43_Bias')))(prelu14_3)
prelu15_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_51_Alpha')), shared_axes=[1, 2])(conv15_1)
depthconv15_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_21_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_21_Bias')))(prelu15_1)

conv15_2 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_44_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_44_Bias')))(depthconv15_1)
prelu15_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_52_Alpha')), shared_axes=[1, 2])(conv15_2)
depthconv15_2 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_22_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_22_Bias')))(prelu15_2)

prelu15_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_53_Alpha')), shared_axes=[1, 2])(depthconv15_2)
conv15_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_45_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_45_Bias')))(prelu15_3)

add15_1 = Add()([conv15_3, prelu14_3])
prelu15_4 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_54_Alpha')), shared_axes=[1, 2])(add15_1)


# Block_16
conv16_1 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_46_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_46_Bias')))(prelu15_4)
prelu16_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_55_Alpha')), shared_axes=[1, 2])(conv16_1)

conv16_2 = Conv2D(filters=8, kernel_size=[3, 3], strides=[1, 1], padding='same', dilation_rate=[8, 8],
                 kernel_initializer=Constant(np.load('weights/conv2d_47_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_47_Bias')))(prelu16_1)
prelu16_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_56_Alpha')), shared_axes=[1, 2])(conv16_2)

conv16_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_48_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_48_Bias')))(prelu16_2)

add16_1 = Add()([conv16_3, prelu15_4])
prelu16_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_57_Alpha')), shared_axes=[1, 2])(add16_1)


# Block_17
conv17_1 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_49_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_49_Bias')))(prelu16_3)
prelu17_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_58_Alpha')), shared_axes=[1, 2])(conv17_1)
depthconv17_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_23_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_23_Bias')))(prelu17_1)

conv17_2 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_50_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_50_Bias')))(depthconv17_1)
prelu17_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_59_Alpha')), shared_axes=[1, 2])(conv17_2)
depthconv17_2 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_24_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_24_Bias')))(prelu17_2)

prelu17_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_60_Alpha')), shared_axes=[1, 2])(depthconv17_2)
conv17_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_51_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_51_Bias')))(prelu17_3)

add17_1 = Add()([conv17_3, prelu16_3])
prelu17_4 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_61_Alpha')), shared_axes=[1, 2])(add17_1)


# Block_18
conv18_1 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_46_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_46_Bias')))(prelu17_4)
prelu18_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_55_Alpha')), shared_axes=[1, 2])(conv18_1)

conv18_2 = Conv2D(filters=8, kernel_size=[3, 3], strides=[1, 1], padding='same', dilation_rate=[2, 2],
                 kernel_initializer=Constant(np.load('weights/conv2d_47_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_47_Bias')))(prelu18_1)
prelu18_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_56_Alpha')), shared_axes=[1, 2])(conv18_2)

conv18_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_48_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_48_Bias')))(prelu18_2)

add18_1 = Add()([conv18_3, prelu17_4])
prelu18_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_57_Alpha')), shared_axes=[1, 2])(add18_1)


# Block_19
conv19_1 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_55_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_55_Bias')))(prelu18_3)
prelu19_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_65_Alpha')), shared_axes=[1, 2])(conv19_1)
depthconv19_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_25_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_25_Bias')))(prelu19_1)

conv19_2 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_56_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_56_Bias')))(depthconv19_1)
prelu19_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_66_Alpha')), shared_axes=[1, 2])(conv19_2)
conv19_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_57_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_57_Bias')))(prelu19_2)

add19_1 = Add()([conv19_3, prelu18_3])
prelu19_4 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_67_Alpha')), shared_axes=[1, 2])(add19_1)


# Block_20
conv20_1 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_58_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_58_Bias')))(prelu19_4)
prelu20_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_68_Alpha')), shared_axes=[1, 2])(conv20_1)

conv20_2 = Conv2D(filters=8, kernel_size=[3, 3], strides=[1, 1], padding='same', dilation_rate=[4, 4],
                 kernel_initializer=Constant(np.load('weights/conv2d_59_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_59_Bias')))(prelu20_1)
prelu20_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_69_Alpha')), shared_axes=[1, 2])(conv20_2)

conv20_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_60_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_60_Bias')))(prelu20_2)

add20_1 = Add()([conv20_3, prelu19_4])
prelu20_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_70_Alpha')), shared_axes=[1, 2])(add20_1)


# Block_21
conv21_1 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_61_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_61_Bias')))(prelu20_3)
prelu21_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_71_Alpha')), shared_axes=[1, 2])(conv21_1)
depthconv21_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_26_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_26_Bias')))(prelu21_1)

conv21_2 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_62_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_62_Bias')))(depthconv21_1)
prelu21_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_72_Alpha')), shared_axes=[1, 2])(conv21_2)
depthconv21_2 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_27_Kernel')),
                 bias_initializer=Constant(np.load('weights/depthwise_conv2d_27_Bias')))(prelu21_2)

prelu21_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_73_Alpha')), shared_axes=[1, 2])(depthconv21_2)
conv21_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_63_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_63_Bias')))(prelu21_3)

add21_1 = Add()([conv21_3, prelu20_3])
prelu21_4 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_74_Alpha')), shared_axes=[1, 2])(add21_1)


# Block_22
conv22_1 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_64_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_64_Bias')))(prelu21_4)
prelu22_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_75_Alpha')), shared_axes=[1, 2])(conv22_1)

conv22_2 = Conv2D(filters=8, kernel_size=[3, 3], strides=[1, 1], padding='same', dilation_rate=[8, 8],
                 kernel_initializer=Constant(np.load('weights/conv2d_65_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_65_Bias')))(prelu22_1)
prelu22_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_76_Alpha')), shared_axes=[1, 2])(conv22_2)

conv22_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_66_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_66_Bias')))(prelu22_2)

add22_1 = Add()([conv22_3, prelu21_4])
prelu22_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_77_Alpha')), shared_axes=[1, 2])(add22_1)


# Block_23
conv23_1 = Conv2D(filters=4, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_67_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_67_Bias')))(prelu22_3)
prelu23_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_78_Alpha')), shared_axes=[1, 2])(conv23_1)

conv23_2 = Conv2D(filters=4, kernel_size=[3, 3], strides=[1, 1], padding='same', dilation_rate=[8, 8],
                 kernel_initializer=Constant(np.load('weights/conv2d_68_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_68_Bias')))(prelu23_1)
prelu23_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_79_Alpha')), shared_axes=[1, 2])(conv23_2)

conv23_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_69_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_69_Bias')))(prelu23_2)

add23_1 = Add()([conv23_3, prelu22_3])
prelu23_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_80_Alpha')), shared_axes=[1, 2])(add23_1)


# Block_24
conv24_1 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_70_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_70_Bias')))(prelu23_3)
prelu24_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_81_Alpha')), shared_axes=[1, 2])(conv24_1)
convtransbias24_1 = Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=(2, 2), padding='same',
                 kernel_initializer=Constant(np.load('weights/conv2d_transpose_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_transpose_Bias')))(prelu24_1)
prelu24_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_82_Alpha')), shared_axes=[1, 2])(convtransbias24_1)
conv24_2 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_71_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_71_Bias')))(prelu24_2)

conv24_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_72_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_72_Bias')))(prelu23_3)
maxunpool24_1 = MaxUnpooling2D(size=[2, 2])([conv24_3, maxpoolarg10_1[1]])

add24_1 = Add()([conv24_2, maxunpool24_1])
prelu24_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_77_Alpha')), shared_axes=[1, 2])(add24_1)

concat24_1 = Concatenate()([prelu24_3, prelu5_4])


# Block_25
conv25_1 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_73_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_73_Bias')))(concat24_1)
prelu25_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_84_Alpha')), shared_axes=[1, 2])(conv25_1)

conv25_2 = Conv2D(filters=8, kernel_size=[3, 3], strides=[1, 1], padding='same', dilation_rate=[8, 8],
                 kernel_initializer=Constant(np.load('weights/conv2d_74_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_74_Bias')))(prelu25_1)
prelu25_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_85_Alpha')), shared_axes=[1, 2])(conv25_2)

conv25_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_75_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_75_Bias')))(prelu25_2)

conv25_4 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_76_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_76_Bias')))(concat24_1)

add25_1 = Add()([conv25_3, conv25_4])
prelu25_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_86_Alpha')), shared_axes=[1, 2])(add25_1)


# Block_26
conv26_1 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_77_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_77_Bias')))(prelu25_3)
prelu26_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_87_Alpha')), shared_axes=[1, 2])(conv26_1)
convtransbias26_1 = Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=(2, 2), padding='same',
                 kernel_initializer=Constant(np.load('weights/conv2d_transpose_1_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_transpose_1_Bias')))(prelu26_1)
prelu26_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_88_Alpha')), shared_axes=[1, 2])(convtransbias26_1)
conv26_2 = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_78_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_78_Bias')))(prelu26_2)

conv26_3 = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_79_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_79_Bias')))(prelu25_3)
maxunpool26_1 = MaxUnpooling2D(size=[2, 2])([conv26_3, maxpoolarg5_1[1]])

add26_1 = Add()([conv26_2, maxunpool26_1])
prelu26_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_89_Alpha')), shared_axes=[1, 2])(add26_1)

concat26_1 = Concatenate()([prelu26_3, prelu2_4])


# Block_27
conv27_1 = Conv2D(filters=4, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_80_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_80_Bias')))(concat26_1)
prelu27_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_90_Alpha')), shared_axes=[1, 2])(conv27_1)

conv27_2 = Conv2D(filters=4, kernel_size=[3, 3], strides=[1, 1], padding='same', dilation_rate=[8, 8],
                 kernel_initializer=Constant(np.load('weights/conv2d_81_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_81_Bias')))(prelu27_1)
prelu27_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_91_Alpha')), shared_axes=[1, 2])(conv27_2)

conv27_3 = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_82_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_82_Bias')))(prelu27_2)

conv27_4 = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_83_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_83_Bias')))(concat26_1)

add27_1 = Add()([conv27_3, conv27_4])
prelu27_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_92_Alpha')), shared_axes=[1, 2])(add27_1)


# Block_28
conv28_1 = Conv2D(filters=4, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_84_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_84_Bias')))(prelu27_3)
prelu28_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_93_Alpha')), shared_axes=[1, 2])(conv28_1)
convtransbias28_1 = Conv2DTranspose(filters=4, kernel_size=(3, 3), strides=(2, 2), padding='same',
                 kernel_initializer=Constant(np.load('weights/conv2d_transpose_2_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_transpose_2_Bias')))(prelu28_1)
prelu28_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_94_Alpha')), shared_axes=[1, 2])(convtransbias28_1)
conv28_2 = Conv2D(filters=32, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_85_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_85_Bias')))(prelu28_2)

conv28_3 = Conv2D(filters=32, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_86_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_86_Bias')))(prelu27_3)
maxunpool28_1 = MaxUnpooling2D(size=[2, 2])([conv28_3, maxpoolarg2_1[1]])

add28_1 = Add()([conv28_2, maxunpool28_1])
prelu28_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_95_Alpha')), shared_axes=[1, 2])(add28_1)


# Block_29
conv29_1 = Conv2D(filters=4, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_87_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_87_Bias')))(prelu28_3)
prelu29_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_96_Alpha')), shared_axes=[1, 2])(conv29_1)

conv29_2 = Conv2D(filters=4, kernel_size=[3, 3], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_88_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_88_Bias')))(prelu29_1)
prelu29_2 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_97_Alpha')), shared_axes=[1, 2])(conv29_2)

conv29_3 = Conv2D(filters=32, kernel_size=[1, 1], strides=[1, 1], padding='same', dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_89_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_89_Bias')))(prelu29_2)

add29_1 = Add()([conv29_3, prelu28_3])
prelu29_3 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_98_Alpha')), shared_axes=[1, 2])(add29_1)


# Block_30
convtransbias30_1 = Conv2DTranspose(filters=8, kernel_size=(2, 2), strides=(2, 2), padding='same',
                 kernel_initializer=Constant(np.load('weights/conv2d_transpose_3_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_transpose_3_Bias')))(prelu29_3)
prelu30_1 = PReLU(alpha_initializer=Constant(np.load('weights/p_re_lu_99_Alpha')), shared_axes=[1, 2])(convtransbias30_1)
convtransbias30_2 = Conv2DTranspose(filters=2, kernel_size=(2, 2), strides=(2, 2), padding='same',
                 kernel_initializer=Constant(np.load('weights/conv2d_transpose_4_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/conv2d_transpose_4_Bias')), name='conv2d_transpose_4')(prelu30_1)


# model = Model(inputs=inputs, outputs=[prelu2_4])
model = Model(inputs=inputs, outputs=[convtransbias30_2])

model.summary()

tf.saved_model.save(model, 'saved_model_{}x{}'.format(height, width))
model.save('hair_segmentation_{}x{}.h5'.format(height, width))

full_model = tf.function(lambda inputs: model(inputs))
full_model = full_model.get_concrete_function(inputs = (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)))
frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
frozen_func.graph.as_graph_def()
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=".",
                    name="hair_segmentation_{}x{}_float32.pb".format(height, width),
                    as_text=False)

# No Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
with open('hair_segmentation_{}x{}_float32.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print("tflite convert complete! - hair_segmentation_{}x{}_float32.tflite".format(height, width))


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
with open('hair_segmentation_{}x{}_weight_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print("Weight Quantization complete! - hair_segmentation_{}x{}_weight_quant.tflite".format(height, width))


# def representative_dataset_gen():
#     for image in raw_test_data:
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
#         image = tf.image.resize(image, (height, width))
#         image = image[np.newaxis,:,:,:]
#         print('image.shape:', image.shape)
#         yield [image]

# raw_test_data = np.load('calibration_data_img_person.npy', allow_pickle=True)


# # Integer Quantization - Input/Output=float32
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('hair_segmentation_{}x{}_integer_quant.tflite'.format(height, width), 'wb') as w:
#     w.write(tflite_quant_model)
# print("Integer Quantization complete! - hair_segmentation_{}x{}_integer_quant.tflite".format(height, width))


# # Full Integer Quantization - Input/Output=int8
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('hair_segmentation_{}x{}_full_integer_quant.tflite'.format(height, width), 'wb') as w:
#     w.write(tflite_quant_model)
# print("Full Integer Quantization complete! - hair_segmentation_{}x{}_full_integer_quant.tflite".format(height, width))


# # Float16 Quantization - Input/Output=float32
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16, tf.lite.OpsSet.SELECT_TF_OPS]
# tflite_quant_model = converter.convert()
# with open('hair_segmentation_{}x{}_float16_quant.tflite'.format(height, width), 'wb') as w:
#     w.write(tflite_quant_model)
# print("Float16 Quantization complete! - hair_segmentation_{}x{}_float16_quant.tflite".format(height, width))


# # EdgeTPU
# import subprocess
# result = subprocess.check_output(["edgetpu_compiler", "-s", "hair_segmentation_{}x{}_full_integer_quant.tflite".format(height, width)])
# print(result)
