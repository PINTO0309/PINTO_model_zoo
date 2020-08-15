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

###  saved_model_cli show --dir saved_model_pose_landmark_upper_body/ --tag_set serve --signature_def serving_default

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, ReLU, MaxPool2D, Reshape, Concatenate, Layer
from tensorflow.keras.initializers import Constant
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
                 kernel_initializer=Constant(np.load('weights2/conv2d_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_Bias')))(inputs)
depthconv1_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_Bias')))(conv1_1)
conv1_2 = Conv2D(filters=24, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_1_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_1_Bias')))(depthconv1_1)
add1_1 = Add()([conv1_1, conv1_2])
relu1_1 = ReLU()(add1_1)

# Block_02
depthconv2_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_1_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_1_Bias')))(relu1_1)
conv2_1 = Conv2D(filters=24, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_2_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_2_Bias')))(depthconv2_1)
add2_1 = Add()([relu1_1, conv2_1])
relu2_1 = ReLU()(add2_1)

# Block_03
depthconv3_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_2_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_2_Bias')))(relu2_1)
conv3_1 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_3_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_3_Bias')))(depthconv3_1)
maxpool3_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(relu2_1)
pad3_1 = tf.pad(maxpool3_1, paddings=tf.constant(np.load('weights2/channel_padding_Paddings')))
add3_1 = Add()([conv3_1, pad3_1])
relu3_1 = ReLU()(add3_1)

# Block_04
depthconv4_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_3_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_3_Bias')))(relu3_1)
conv4_1 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_4_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_4_Bias')))(depthconv4_1)
add4_1 = Add()([relu3_1, conv4_1])
relu4_1 = ReLU()(add4_1)

# Block_05
depthconv5_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_4_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_4_Bias')))(relu4_1)
conv5_1 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_5_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_5_Bias')))(depthconv5_1)
add5_1 = Add()([relu4_1, conv5_1])
relu5_1 = ReLU()(add5_1)

# Block_06
depthconv6_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_5_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_5_Bias')))(relu5_1)
conv6_1 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_6_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_6_Bias')))(depthconv6_1)
add6_1 = Add()([relu5_1, conv6_1])
relu6_1 = ReLU()(add6_1)

# Block_07
depthconv7_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_6_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_6_Bias')))(relu6_1)
conv7_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_7_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_7_Bias')))(depthconv7_1)
maxpool7_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(relu6_1)
pad7_1 = tf.pad(maxpool7_1, paddings=tf.constant(np.load('weights2/channel_padding_1_Paddings')))
add7_1 = Add()([conv7_1, pad7_1])
relu7_1 = ReLU()(add7_1)

# Block_08
depthconv8_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_7_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_7_Bias')))(relu7_1)
conv8_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_8_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_8_Bias')))(depthconv8_1)
add8_1 = Add()([relu7_1, conv8_1])
relu8_1 = ReLU()(add8_1)

# Block_09
depthconv9_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_8_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_8_Bias')))(relu8_1)
conv9_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_9_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_9_Bias')))(depthconv9_1)
add9_1 = Add()([relu8_1, conv9_1])
relu9_1 = ReLU()(add9_1)

# Block_10
depthconv10_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_9_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_9_Bias')))(relu9_1)
conv10_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_10_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_10_Bias')))(depthconv10_1)
add10_1 = Add()([relu9_1, conv10_1])
relu10_1 = ReLU()(add10_1)

# Block_11
depthconv11_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_10_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_10_Bias')))(relu10_1)
conv11_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_11_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_11_Bias')))(depthconv11_1)
add11_1 = Add()([relu10_1, conv11_1])
relu11_1 = ReLU()(add11_1)

# Block_12
depthconv12_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_11_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_11_Bias')))(relu11_1)
conv12_1 = Conv2D(filters=192, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_12_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_12_Bias')))(depthconv12_1)
maxpool12_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(relu11_1)
pad12_1 = tf.pad(maxpool12_1, paddings=tf.constant(np.load('weights2/channel_padding_2_Paddings')))
add12_1 = Add()([conv12_1, pad12_1])
relu12_1 = ReLU()(add12_1)

# Block_13
depthconv13_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_12_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_12_Bias')))(relu12_1)
conv13_1 = Conv2D(filters=192, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_13_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_13_Bias')))(depthconv13_1)
add13_1 = Add()([relu12_1, conv13_1])
relu13_1 = ReLU()(add13_1)

# Block_14
depthconv14_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_13_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_13_Bias')))(relu13_1)
conv14_1 = Conv2D(filters=192, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_14_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_14_Bias')))(depthconv14_1)
add14_1 = Add()([relu13_1, conv14_1])
relu14_1 = ReLU()(add14_1)

# Block_15
depthconv15_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_14_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_14_Bias')))(relu14_1)
conv15_1 = Conv2D(filters=192, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_15_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_15_Bias')))(depthconv15_1)
add15_1 = Add()([relu14_1, conv15_1])
relu15_1 = ReLU()(add15_1)

# Block_16
depthconv16_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_15_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_15_Bias')))(relu15_1)
conv16_1 = Conv2D(filters=192, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_16_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_16_Bias')))(depthconv16_1)
add16_1 = Add()([relu15_1, conv16_1])
relu16_1 = ReLU()(add16_1)

# Block_17
depthconv17_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_16_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_16_Bias')))(relu16_1)
conv17_1 = Conv2D(filters=192, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_17_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_17_Bias')))(depthconv17_1)
add17_1 = Add()([relu16_1, conv17_1])
relu17_1 = ReLU()(add17_1)

# Block_18
depthconv18_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_17_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_17_Bias')))(relu17_1)
conv18_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_18_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_18_Bias')))(depthconv18_1)
maxpool18_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(relu17_1)
pad18_1 = tf.pad(maxpool18_1, paddings=tf.constant(np.load('weights2/channel_padding_3_Paddings')))
add18_1 = Add()([conv18_1, pad18_1])
relu18_1 = ReLU()(add18_1)

# Block_19
depthconv19_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_18_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_18_Bias')))(relu18_1)
conv19_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_19_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_19_Bias')))(depthconv19_1)
add19_1 = Add()([relu18_1, conv19_1])
relu19_1 = ReLU()(add19_1)

# Block_20
depthconv20_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_19_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_19_Bias')))(relu19_1)
conv20_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_20_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_20_Bias')))(depthconv20_1)
add20_1 = Add()([relu19_1, conv20_1])
relu20_1 = ReLU()(add20_1)

# Block_21
depthconv21_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_20_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_20_Bias')))(relu20_1)
conv21_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_21_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_21_Bias')))(depthconv21_1)
add21_1 = Add()([relu20_1, conv21_1])
relu21_1 = ReLU()(add21_1)

# Block_22
depthconv22_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_21_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_21_Bias')))(relu21_1)
conv22_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_22_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_22_Bias')))(depthconv22_1)
add22_1 = Add()([relu21_1, conv22_1])
relu22_1 = ReLU()(add22_1)

# Block_23
depthconv23_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_22_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_22_Bias')))(relu22_1)
conv23_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_23_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_23_Bias')))(depthconv23_1)
add23_1 = Add()([relu22_1, conv23_1])
relu23_1 = ReLU()(add23_1)

# Block_24
depthconv24_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_23_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_23_Bias')))(relu23_1)
conv24_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_24_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_24_Bias')))(depthconv24_1)
add24_1 = Add()([relu23_1, conv24_1])
relu24_1 = ReLU()(add24_1)

# Block_25
depthconv25_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_24_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_24_Bias')))(relu24_1)
conv25_1 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights2/conv2d_25_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_25_Bias')))(depthconv25_1)
resize25_1 = tf.image.resize(conv25_1, np.load('weights2/up_sampling2d_Size'))

depthconv25_2 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_25_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_25_Bias')))(relu17_1)
conv25_2 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights2/conv2d_26_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_26_Bias')))(depthconv25_2)

add25_1 = Add()([resize25_1, conv25_2])

resize25_2 = tf.image.resize(add25_1, np.load('weights2/up_sampling2d_1_Size'))

depthconv25_3 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_26_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_26_Bias')))(relu11_1)
conv25_3 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights2/conv2d_27_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_27_Bias')))(depthconv25_3)

add25_2 = Add()([resize25_2, conv25_3])

resize25_3 = tf.image.resize(add25_2, np.load('weights2/up_sampling2d_2_Size'))

depthconv25_4 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_27_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_27_Bias')))(relu6_1)
conv25_4 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights2/conv2d_28_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_28_Bias')))(depthconv25_4)

add25_3 = Add()([resize25_3, conv25_4])

# Block_26
depthconv26_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_28_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_28_Bias')))(add25_3)
conv26_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_29_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_29_Bias')))(depthconv26_1)
maxpool26_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(add25_3)
pad26_1 = tf.pad(maxpool26_1, paddings=tf.constant(np.load('weights2/channel_padding_4_Paddings')))
add26_1 = Add()([conv26_1, pad26_1])
relu26_1 = ReLU()(add26_1)

# Block_27
depthconv27_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_29_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_29_Bias')))(relu26_1)
conv27_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_30_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_30_Bias')))(depthconv27_1)
add27_1 = Add()([relu26_1, conv27_1])
relu27_1 = ReLU()(add27_1)

# Block_28
depthconv28_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_30_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_30_Bias')))(relu27_1)
conv28_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_31_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_31_Bias')))(depthconv28_1)
add28_1 = Add()([relu27_1, conv28_1])
relu28_1 = ReLU()(add28_1)

# Block_29
depthconv29_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_31_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_31_Bias')))(relu28_1)
conv29_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_32_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_32_Bias')))(depthconv29_1)
add29_1 = Add()([relu28_1, conv29_1])
relu29_1 = ReLU()(add29_1)

# Block_30
depthconv30_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_32_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_32_Bias')))(relu29_1)
conv30_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_33_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_33_Bias')))(depthconv30_1)
add30_1 = Add()([relu29_1, conv30_1])
relu30_1 = ReLU()(add30_1)

# Block_31
depthconv31_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_33_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_33_Bias')))(relu11_1)
conv31_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights2/conv2d_34_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_34_Bias')))(depthconv31_1)
add31_1 = Add()([relu30_1, conv31_1])

# Block_32
depthconv32_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_34_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_34_Bias')))(add31_1)
conv32_1 = Conv2D(filters=192, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_35_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_35_Bias')))(depthconv32_1)
maxpool32_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(add31_1)
pad32_1 = tf.pad(maxpool32_1, paddings=tf.constant(np.load('weights2/channel_padding_5_Paddings')))
add32_1 = Add()([conv32_1, pad32_1])
relu32_1 = ReLU()(add32_1)

# Block_33
depthconv33_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_35_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_35_Bias')))(relu32_1)
conv33_1 = Conv2D(filters=192, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_36_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_36_Bias')))(depthconv33_1)
add33_1 = Add()([relu32_1, conv33_1])
relu33_1 = ReLU()(add33_1)

# Block_34
depthconv34_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_36_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_36_Bias')))(relu33_1)
conv34_1 = Conv2D(filters=192, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_37_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_37_Bias')))(depthconv34_1)
add34_1 = Add()([relu33_1, conv34_1])
relu34_1 = ReLU()(add34_1)

# Block_35
depthconv35_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_37_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_37_Bias')))(relu34_1)
conv35_1 = Conv2D(filters=192, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_38_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_38_Bias')))(depthconv35_1)
add35_1 = Add()([relu34_1, conv35_1])
relu35_1 = ReLU()(add35_1)

# Block_36
depthconv36_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_38_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_38_Bias')))(relu35_1)
conv36_1 = Conv2D(filters=192, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_39_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_39_Bias')))(depthconv36_1)
add36_1 = Add()([relu35_1, conv36_1])
relu36_1 = ReLU()(add36_1)

# Block_37
depthconv37_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_39_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_39_Bias')))(relu36_1)
conv37_1 = Conv2D(filters=192, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_40_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_40_Bias')))(depthconv37_1)
add37_1 = Add()([relu36_1, conv37_1])
relu37_1 = ReLU()(add37_1)

# Block_38
depthconv38_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_40_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_40_Bias')))(relu17_1)
conv38_1 = Conv2D(filters=192, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights2/conv2d_41_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_41_Bias')))(depthconv38_1)
add38_1 = Add()([conv38_1, relu37_1])

# Block_39
depthconv39_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_41_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_41_Bias')))(add38_1)
conv39_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_42_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_42_Bias')))(depthconv39_1)
maxpool39_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(add38_1)
pad39_1 = tf.pad(maxpool39_1, paddings=tf.constant(np.load('weights2/channel_padding_6_Paddings')))
add39_1 = Add()([conv39_1, pad39_1])
relu39_1 = ReLU()(add39_1)

# Block_40
depthconv40_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_42_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_42_Bias')))(relu39_1)
conv40_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_43_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_43_Bias')))(depthconv40_1)
add40_1 = Add()([relu39_1, conv40_1])
relu40_1 = ReLU()(add40_1)

# Block_41
depthconv41_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_43_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_43_Bias')))(relu40_1)
conv41_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_44_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_44_Bias')))(depthconv41_1)
add41_1 = Add()([relu40_1, conv41_1])
relu41_1 = ReLU()(add41_1)

# Block_42
depthconv42_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_44_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_44_Bias')))(relu41_1)
conv42_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_45_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_45_Bias')))(depthconv42_1)
add42_1 = Add()([relu41_1, conv42_1])
relu42_1 = ReLU()(add42_1)

# Block_43
depthconv43_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_45_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_45_Bias')))(relu42_1)
conv43_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_46_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_46_Bias')))(depthconv43_1)
add43_1 = Add()([relu42_1, conv43_1])
relu43_1 = ReLU()(add43_1)

# Block_44
depthconv44_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_46_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_46_Bias')))(relu43_1)
conv44_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_47_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_47_Bias')))(depthconv44_1)
add44_1 = Add()([relu43_1, conv44_1])
relu44_1 = ReLU()(add44_1)

# Block_45
depthconv45_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_47_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_47_Bias')))(relu44_1)
conv45_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_48_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_48_Bias')))(depthconv45_1)
add45_1 = Add()([relu44_1, conv45_1])
relu45_1 = ReLU()(add45_1)

# Block_46
depthconv46_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_48_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_48_Bias')))(relu24_1)
conv46_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights2/conv2d_49_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_49_Bias')))(depthconv46_1)
add46_1 = Add()([conv46_1, relu45_1])

# Block_47
depthconv47_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_49_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_49_Bias')))(add46_1)
conv47_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_50_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_50_Bias')))(depthconv47_1)
maxpool47_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(add46_1)
add47_1 = Add()([conv47_1, maxpool47_1])
relu47_1 = ReLU()(add47_1)

# Block_48
depthconv48_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_50_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_50_Bias')))(relu47_1)
conv48_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_51_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_51_Bias')))(depthconv48_1)
add48_1 = Add()([conv48_1, relu47_1])
relu48_1 = ReLU()(add48_1)

# Block_49
depthconv49_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_51_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_51_Bias')))(relu48_1)
conv49_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_52_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_52_Bias')))(depthconv49_1)
add49_1 = Add()([conv49_1, relu48_1])
relu49_1 = ReLU()(add49_1)

# Block_50
depthconv50_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_52_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_52_Bias')))(relu49_1)
conv50_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_53_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_53_Bias')))(depthconv50_1)
add50_1 = Add()([conv50_1, relu49_1])
relu50_1 = ReLU()(add50_1)

# Block_51
depthconv51_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_53_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_53_Bias')))(relu50_1)
conv51_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_54_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_54_Bias')))(depthconv51_1)
add51_1 = Add()([conv51_1, relu50_1])
relu51_1 = ReLU()(add51_1)

# Block_52
depthconv52_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_54_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_54_Bias')))(relu51_1)
conv52_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_55_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_55_Bias')))(depthconv52_1)
add52_1 = Add()([conv52_1, relu51_1])
relu52_1 = ReLU()(add52_1)

# Block_53
depthconv53_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_55_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_55_Bias')))(relu52_1)
conv53_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_56_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_56_Bias')))(depthconv53_1)
add53_1 = Add()([conv53_1, relu52_1])
relu53_1 = ReLU()(add53_1)

# Block_54
depthconv54_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_56_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_56_Bias')))(relu53_1)
conv54_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_57_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_57_Bias')))(depthconv54_1)
add54_1 = Add()([conv54_1, relu53_1])
relu54_1 = ReLU()(add54_1)

# Block_55
depthconv55_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_57_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_57_Bias')))(relu54_1)
conv55_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_58_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_58_Bias')))(depthconv55_1)
maxpool55_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(relu54_1)
add55_1 = Add()([conv55_1, maxpool55_1])
relu55_1 = ReLU()(add55_1)

# Block_56
depthconv56_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_58_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_58_Bias')))(relu55_1)
conv56_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_59_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_59_Bias')))(depthconv56_1)
add56_1 = Add()([conv56_1, relu55_1])
relu56_1 = ReLU()(add56_1)

# Block_57
depthconv57_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_59_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_59_Bias')))(relu56_1)
conv57_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_60_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_60_Bias')))(depthconv57_1)
add57_1 = Add()([conv57_1, relu56_1])
relu57_1 = ReLU()(add57_1)

# Block_58
depthconv58_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_60_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_60_Bias')))(relu57_1)
conv58_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_61_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_61_Bias')))(depthconv58_1)
add58_1 = Add()([conv58_1, relu57_1])
relu58_1 = ReLU()(add58_1)

# Block_59
depthconv59_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_61_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_61_Bias')))(relu58_1)
conv59_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_62_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_62_Bias')))(depthconv59_1)
add59_1 = Add()([conv59_1, relu58_1])
relu59_1 = ReLU()(add59_1)

# Block_60
depthconv60_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_62_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_62_Bias')))(relu59_1)
conv60_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_63_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_63_Bias')))(depthconv60_1)
add60_1 = Add()([conv60_1, relu59_1])
relu60_1 = ReLU()(add60_1)

# Block_61
depthconv61_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_63_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_63_Bias')))(relu60_1)
conv61_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_64_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_64_Bias')))(depthconv61_1)
add61_1 = Add()([conv61_1, relu60_1])
relu61_1 = ReLU()(add61_1)

# Block_62
depthconv62_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_64_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_64_Bias')))(relu61_1)
conv62_1 = Conv2D(filters=288, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv2d_65_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_65_Bias')))(depthconv62_1)
add62_1 = Add()([conv62_1, relu61_1])
relu62_1 = ReLU()(add62_1)

# Block_63
depthconv63_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_66_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_66_Bias')))(add25_3)
conv63_1 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights2/conv2d_67_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_67_Bias')))(depthconv63_1)
resize63_1 = tf.image.resize(conv63_1, np.load('weights2/up_sampling2d_3_Size'))

depthconv63_2 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_65_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_65_Bias')))(relu2_1)
conv63_2 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights2/conv2d_66_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_66_Bias')))(depthconv63_2)

add63_1 = Add()([resize63_1, conv63_2])

depthconv63_3 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights2/depthwise_conv2d_67_Kernel')),
                 bias_initializer=Constant(np.load('weights2/depthwise_conv2d_67_Bias')))(add63_1)
conv63_3 = Conv2D(filters=8, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights2/conv2d_68_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv2d_68_Bias')))(depthconv63_3)

# Final Block_99
conv99_1 = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/output_segmentation_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/output_segmentation_Bias')), name='output_segmentation')(conv63_3)

conv99_2 = Conv2D(filters=1, kernel_size=[2, 2], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/conv_poseflag_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/conv_poseflag_Bias')))(relu62_1)
sigm99_1 = tf.math.sigmoid(conv99_2, name='output_poseflag')
# reshape99_1 = tf.reshape(sigm99_1, (1, 1), name='output_poseflag')

conv99_3 = Conv2D(filters=124, kernel_size=[2, 2], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights2/convld_3d_Kernel').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights2/convld_3d_Bias')))(relu62_1)
reshape99_2 = tf.reshape(conv99_3, (1, 124), name='ld_3d')



model = Model(inputs=inputs, outputs=[conv99_1, sigm99_1, reshape99_2])

model.summary()

tf.saved_model.save(model, 'saved_model_pose_landmark_upper_body')
model.save('pose_landmark_upper_body.h5')


# No Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('pose_landmark_upper_body_256x256_float32.tflite', 'wb') as w:
    w.write(tflite_model)
print("tflite convert complete! - pose_landmark_upper_body_256x256_float32.tflite")


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
with open('pose_landmark_upper_body_256x256_weight_quant.tflite', 'wb') as w:
    w.write(tflite_model)
print("Weight Quantization complete! - pose_landmark_upper_body_256x256_weight_quant.tflite")


def representative_dataset_gen():
    for image in raw_test_data:
        image = tf.image.resize(image, (256, 256))
        image = image[np.newaxis,:,:,:]
        yield [image]

raw_test_data = np.load('calibration_data_img_person.npy', allow_pickle=True)


# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('pose_landmark_upper_body_256x256_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - pose_landmark_upper_body_256x256_integer_quant.tflite")


# Full Integer Quantization - Input/Output=int8
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('pose_landmark_upper_body_256x256_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Full Integer Quantization complete! - pose_landmark_upper_body_256x256_full_integer_quant.tflite")


# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('pose_landmark_upper_body_256x256_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - pose_landmark_upper_body_256x256_float16_quant.tflite")


# EdgeTPU
import subprocess
result = subprocess.check_output(["edgetpu_compiler", "-s", "pose_landmark_upper_body_256x256_full_integer_quant.tflite"])
print(result)
