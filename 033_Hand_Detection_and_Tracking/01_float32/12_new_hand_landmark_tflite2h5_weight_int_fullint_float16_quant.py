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

### How to initialize a convolution layer with an arbitrary kernel in Keras? https://stackoverrun.com/ja/q/12269118

###  saved_model_cli show --dir saved_model_hand_landmark_new/ --tag_set serve --signature_def serving_default

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, ReLU, MaxPool2D, Reshape, Concatenate, Layer
from tensorflow.keras.initializers import Constant
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import sys

# tmp = np.load('weights/depthwise_conv2d_Kernel')
# print(tmp.shape)
# print(tmp)

# def init_f(shape, dtype=None):
#        ker = np.load('weights/depthwise_conv2d_Kernel')
#        print(shape)
#        return ker

# sys.exit(0)

inputs = Input(shape=(256, 256, 3), name='input')

# Block_01
conv1_1 = Conv2D(filters=24, kernel_size=[3, 3], strides=[2, 2], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights_new/conv2d_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_Bias')))(inputs)
depthconv1_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_Bias')))(conv1_1)
conv1_2 = Conv2D(filters=24, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_1_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_1_Bias')))(depthconv1_1)
add1_1 = Add()([conv1_1, conv1_2])
relu1_1 = ReLU()(add1_1)

# Block_02
depthconv2_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_1_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_1_Bias')))(relu1_1)
conv2_1 = Conv2D(filters=24, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_2_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_2_Bias')))(depthconv2_1)
add2_1 = Add()([relu1_1, conv2_1])
relu2_1 = ReLU()(add2_1)

# Block_03
depthconv3_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_2_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_2_Bias')))(relu2_1)
conv3_1 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_3_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_3_Bias')))(depthconv3_1)
maxpool3_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(relu2_1)
pad3_1 = tf.pad(maxpool3_1, paddings=tf.constant(np.load('weights_new/channel_padding_Paddings')))

add3_1 = Add()([conv3_1, pad3_1])
relu3_1 = ReLU()(add3_1)

# Block_04
depthconv4_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_3_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_3_Bias')))(relu3_1)
conv4_1 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_4_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_4_Bias')))(depthconv4_1)
add4_1 = Add()([conv4_1, relu3_1])
relu4_1 = ReLU()(add4_1)

# Block_05
depthconv5_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_4_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_4_Bias')))(relu4_1)
conv5_1 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_5_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_5_Bias')))(depthconv5_1)
add5_1 = Add()([relu4_1, conv5_1])
relu5_1 = ReLU()(add5_1)

# Block_06
depthconv6_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_5_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_5_Bias')))(relu5_1)
conv6_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_6_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_6_Bias')))(depthconv6_1)
maxpool6_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(relu5_1)
pad6_1 = tf.pad(maxpool6_1, paddings=tf.constant(np.load('weights_new/channel_padding_1_Paddings')))
add6_1 = Add()([conv6_1, pad6_1])
relu6_1 = ReLU()(add6_1)

# Block_07
depthconv7_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_6_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_6_Bias')))(relu6_1)
conv7_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_7_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_7_Bias')))(depthconv7_1)
add7_1 = Add()([conv7_1, relu6_1])
relu7_1 = ReLU()(add7_1)

# Block_08
depthconv8_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_7_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_7_Bias')))(relu7_1)
conv8_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_8_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_8_Bias')))(depthconv8_1)
add8_1 = Add()([conv8_1, relu7_1])
relu8_1 = ReLU()(add8_1)

# Block_09
depthconv9_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_8_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_8_Bias')))(relu8_1)
conv9_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_9_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_9_Bias')))(depthconv9_1)
maxpool9_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(relu8_1)
add9_1 = Add()([conv9_1, maxpool9_1])
relu9_1 = ReLU()(add9_1)

# Block_10
depthconv10_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_9_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_9_Bias')))(relu9_1)
conv10_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_10_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_10_Bias')))(depthconv10_1)
add10_1 = Add()([conv10_1, relu9_1])
relu10_1 = ReLU()(add10_1)

# Block_11
depthconv11_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_10_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_10_Bias')))(relu10_1)
conv11_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_11_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_11_Bias')))(depthconv11_1)
add11_1 = Add()([conv11_1, relu10_1])
relu11_1 = ReLU()(add11_1)

# Block_12
depthconv12_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_11_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_11_Bias')))(relu11_1)
conv12_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_12_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_12_Bias')))(depthconv12_1)
maxpool12_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(relu11_1)
add12_1 = Add()([conv12_1, maxpool12_1])
relu12_1 = ReLU()(add12_1)
resize12_1 = tf.image.resize(relu12_1, np.load('weights_new/up_sampling2d_Size'))
add12_2 = Add()([relu9_1, resize12_1])

# Block_13
depthconv13_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_12_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_12_Bias')))(add12_2)
conv13_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_13_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_13_Bias')))(depthconv13_1)
add13_1 = Add()([add12_2, conv13_1])
relu13_1 = ReLU()(add13_1)
resize13_1 = tf.image.resize(relu13_1, np.load('weights_new/up_sampling2d_1_Size'))
add13_2 = Add()([relu6_1, resize13_1])

# Block_14
depthconv14_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_13_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_13_Bias')))(add13_2)
conv14_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_14_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_14_Bias')))(depthconv14_1)
add14_1 = Add()([add13_2, conv14_1])
relu14_1 = ReLU()(add14_1)
conv14_2 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_15_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_15_Bias')))(relu14_1)
resize14_1 = tf.image.resize(conv14_2, np.load('weights_new/up_sampling2d_2_Size'))
relu14_2 = ReLU()(relu3_1)
add14_1 = Add()([relu14_2, resize14_1])

# Block_15
depthconv15_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_14_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_14_Bias')))(add14_1)
conv15_1 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_16_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_16_Bias')))(depthconv15_1)
add15_1 = Add()([add14_1, conv15_1])
relu15_1 = ReLU()(add15_1)

# Block_16
depthconv16_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_15_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_15_Bias')))(relu15_1)
conv16_1 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_17_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_17_Bias')))(depthconv16_1)
add16_1 = Add()([conv16_1, relu15_1])
relu16_1 = ReLU()(add16_1)

# Block_17
depthconv17_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_16_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_16_Bias')))(relu16_1)
conv17_1 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_18_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_18_Bias')))(depthconv17_1)
add17_1 = Add()([conv17_1, relu16_1])
relu17_1 = ReLU()(add17_1)

# Block_18
depthconv18_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_17_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_17_Bias')))(relu17_1)
conv18_1 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_19_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_19_Bias')))(depthconv18_1)
add18_1 = Add()([conv18_1, relu17_1])
relu18_1 = ReLU()(add18_1)

# Block_19
depthconv19_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_18_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_18_Bias')))(relu18_1)
conv19_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_20_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_20_Bias')))(depthconv19_1)
maxpool19_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(relu18_1)
pad19_1 = tf.pad(maxpool19_1, paddings=tf.constant(np.load('weights_new/channel_padding_2_Paddings')))
add19_1 = Add()([conv19_1, pad19_1])
relu19_1 = ReLU()(add19_1)

# Block_20
depthconv20_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_19_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_19_Bias')))(relu19_1)
conv20_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_21_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_21_Bias')))(depthconv20_1)
add20_1 = Add()([relu19_1, conv20_1])
relu20_1 = ReLU()(add20_1)

# Block_21
depthconv21_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_20_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_20_Bias')))(relu20_1)
conv21_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_22_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_22_Bias')))(depthconv21_1)
add21_1 = Add()([relu20_1, conv21_1])
relu21_1 = ReLU()(add21_1)

# Block_22
depthconv22_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_21_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_21_Bias')))(relu21_1)
conv22_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_23_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_23_Bias')))(depthconv22_1)
add22_1 = Add()([conv22_1, relu21_1])
relu22_1 = ReLU()(add22_1)

# Block_23
depthconv23_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_22_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_22_Bias')))(relu22_1)
conv23_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_24_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_24_Bias')))(depthconv23_1)
add23_1 = Add()([conv23_1, relu22_1])
relu23_1 = ReLU()(add23_1)

# Block_24
depthconv24_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_23_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_23_Bias')))(relu23_1)
conv24_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_25_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_25_Bias')))(depthconv24_1)
maxpool24_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(relu23_1)
pad24_1 = tf.pad(maxpool24_1, paddings=tf.constant(np.load('weights_new/channel_padding_3_Paddings')))
add24_1 = Add()([conv24_1, pad24_1])
relu24_1 = ReLU()(add24_1)

# Block_25
depthconv25_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_24_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_24_Bias')))(relu24_1)
conv25_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_26_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_26_Bias')))(depthconv25_1)
add25_1 = Add()([relu24_1, conv25_1])
relu25_1 = ReLU()(add25_1)

# Block_26
depthconv26_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_25_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_25_Bias')))(relu25_1)
conv26_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_27_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_27_Bias')))(depthconv26_1)
add26_1 = Add()([relu25_1, conv26_1])
relu26_1 = ReLU()(add26_1)

# Block_27
depthconv27_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_26_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_26_Bias')))(relu26_1)
conv27_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_28_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_28_Bias')))(depthconv27_1)
add27_1 = Add()([conv27_1, relu26_1])
relu27_1 = ReLU()(add27_1)

# Block_28
depthconv28_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_27_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_27_Bias')))(relu27_1)
conv28_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_29_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_29_Bias')))(depthconv28_1)
add28_1 = Add()([conv28_1, relu27_1])
relu28_1 = ReLU()(add28_1)

# Block_29
depthconv29_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_28_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_28_Bias')))(relu28_1)
conv29_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_30_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_30_Bias')))(depthconv29_1)
maxpool29_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(relu28_1)
add29_1 = Add()([conv29_1, maxpool29_1])
relu29_1 = ReLU()(add29_1)

# Block_30
depthconv30_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_29_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_29_Bias')))(relu29_1)
conv30_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_31_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_31_Bias')))(depthconv30_1)
add30_1 = Add()([conv30_1, relu29_1])
relu30_1 = ReLU()(add30_1)

# Block_31
depthconv31_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_30_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_30_Bias')))(relu30_1)
conv31_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_32_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_32_Bias')))(depthconv31_1)
add31_1 = Add()([conv31_1, relu30_1])
relu31_1 = ReLU()(add31_1)

# Block_32
depthconv32_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_31_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_31_Bias')))(relu31_1)
conv32_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_33_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_33_Bias')))(depthconv32_1)
add32_1 = Add()([conv32_1, relu31_1])
relu32_1 = ReLU()(add32_1)

# Block_33
depthconv33_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_32_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_32_Bias')))(relu32_1)
conv33_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_34_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_34_Bias')))(depthconv33_1)
add33_1 = Add()([conv33_1, relu32_1])
relu33_1 = ReLU()(add33_1)

# Block_34
depthconv34_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_33_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_33_Bias')))(relu33_1)
conv34_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_35_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_35_Bias')))(depthconv34_1)
maxpool34_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(relu33_1)
add34_1 = Add()([conv34_1, maxpool34_1])
relu34_1 = ReLU()(add34_1)

# Block_35
depthconv35_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_34_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_34_Bias')))(relu34_1)
conv35_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_36_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_36_Bias')))(depthconv35_1)
add35_1 = Add()([conv35_1, relu34_1])
relu35_1 = ReLU()(add35_1)

# Block_36
depthconv36_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_35_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_35_Bias')))(relu35_1)
conv36_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_37_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_37_Bias')))(depthconv36_1)
add36_1 = Add()([conv36_1, relu35_1])
relu36_1 = ReLU()(add36_1)

# Block_37
depthconv37_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_36_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_36_Bias')))(relu36_1)
conv37_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_38_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_38_Bias')))(depthconv37_1)
add37_1 = Add()([conv37_1, relu36_1])
relu37_1 = ReLU()(add37_1)

# Block_38
depthconv38_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_37_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_37_Bias')))(relu37_1)
conv38_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_39_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_39_Bias')))(depthconv38_1)
add38_1 = Add()([conv38_1, relu37_1])
relu38_1 = ReLU()(add38_1)

# Block_39
depthconv39_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_38_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_38_Bias')))(relu38_1)
conv39_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_40_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_40_Bias')))(depthconv39_1)
maxpool39_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(relu38_1)
add39_1 = Add()([conv39_1, maxpool39_1])
relu39_1 = ReLU()(add39_1)

# Block_40
depthconv40_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_39_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_39_Bias')))(relu39_1)
conv40_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_41_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_41_Bias')))(depthconv40_1)
add40_1 = Add()([conv40_1, relu39_1])
relu40_1 = ReLU()(add40_1)

# Block_41
depthconv41_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_40_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_40_Bias')))(relu40_1)
conv41_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_42_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_42_Bias')))(depthconv41_1)
add41_1 = Add()([conv41_1, relu40_1])
relu41_1 = ReLU()(add41_1)

# Block_42
depthconv42_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_41_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_41_Bias')))(relu41_1)
conv42_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_43_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_43_Bias')))(depthconv42_1)
add42_1 = Add()([conv42_1, relu41_1])
relu42_1 = ReLU()(add42_1)

# Block_43
depthconv43_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_new/depthwise_conv2d_42_Kernel')),
                 bias_initializer=Constant(np.load('weights_new/depthwise_conv2d_42_Bias')))(relu42_1)
conv43_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv2d_44_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv2d_44_Bias')))(depthconv43_1)
add43_1 = Add()([conv43_1, relu42_1])
relu43_1 = ReLU()(add43_1)

# Final Block_99
conv99_1 = Conv2D(filters=1, kernel_size=[2, 2], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv_handflag_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv_handflag_Bias')))(relu43_1)
sigm99_1 = tf.math.sigmoid(conv99_1, name='output_handflag')
# reshape99_1 = tf.reshape(sigm99_1, (1, -1), name='output_handflag')

conv99_2 = Conv2D(filters=1, kernel_size=[2, 2], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/conv_handedness_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/conv_handedness_Bias')))(relu43_1)
sigm99_2 = tf.math.sigmoid(conv99_2, name='output_handedness')
# reshape99_2 = tf.reshape(sigm99_2, (1, -1), name='output_handedness')

conv99_3 = Conv2D(filters=63, kernel_size=[2, 2], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_new/convld_21_3d_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_new/convld_21_3d_Bias')))(relu43_1)
reshape99_3 = tf.reshape(conv99_3, (1, 63), name='ld_21_3d')



model = Model(inputs=inputs, outputs=[sigm99_1, sigm99_2, reshape99_3])
# model = Model(inputs=inputs, outputs=[reshape99_1, reshape99_2, reshape99_3])

model.summary()

tf.saved_model.save(model, 'saved_model_hand_landmark_new')
model.save('hand_landmark_new.h5')

full_model = tf.function(lambda inputs: model(inputs))
full_model = full_model.get_concrete_function(inputs = (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)))
frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
frozen_func.graph.as_graph_def()
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=".",
                    name="hand_landmark_new_256x256_float32.pb",
                    as_text=False)

# No Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('hand_landmark_new_256x256_float32.tflite', 'wb') as w:
    w.write(tflite_model)
print("tflite convert complete! - hand_landmark_new_256x256_float32.tflite")


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
with open('hand_landmark_new_256x256_weight_quant.tflite', 'wb') as w:
    w.write(tflite_model)
print("Weight Quantization complete! - hand_landmark_new_256x256_weight_quant.tflite")


def representative_dataset_gen():
  for data in raw_test_data.take(100):
    image = data['image'].numpy()
    image = tf.image.resize(image, (256, 256))
    image = image[np.newaxis,:,:,:]
    image = image - 127.5
    image = image * 0.007843
    yield [image]

raw_test_data, info = tfds.load(name="the300w_lp", with_info=True, split="train", data_dir="~/TFDS", download=True)


# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('hand_landmark_new_256x256_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - hand_landmark_new_256x256_integer_quant.tflite")


# # Full Integer Quantization - Input/Output=int8
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('hand_landmark_new_256x256_full_integer_quant.tflite', 'wb') as w:
#     w.write(tflite_quant_model)
# print("Full Integer Quantization complete! - hand_landmark_new_256x256_full_integer_quant.tflite")


# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('hand_landmark_new_256x256_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - hand_landmark_new_256x256_float16_quant.tflite")


# # EdgeTPU
# import subprocess
# result = subprocess.check_output(["edgetpu_compiler", "-s", "hand_landmark_new_256x256_full_integer_quant.tflite"])
# print(result)
