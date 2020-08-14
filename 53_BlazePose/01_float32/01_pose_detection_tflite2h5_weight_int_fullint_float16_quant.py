### tensorflow==2.3.0

### https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html
### https://google.github.io/mediapipe/solutions/pose

### https://www.tensorflow.org/api_docs/python/tf/keras/Model
### op types: ['CONV_2D', 'RELU', 'DEPTHWISE_CONV_2D', 'ADD', 'MAX_POOL_2D', 'RESHAPE', 'CONCATENATION']

### https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/Add
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/Concatenate
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer

### How to initialize a convolution layer with an arbitrary kernel in Keras? https://stackoverrun.com/ja/q/12269118

###  saved_model_cli show --dir saved_model_pose_detection/ --tag_set serve --signature_def serving_default

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

inputs = Input(shape=(128, 128, 3), name='input')

# Bolock_01
conv1_1 = Conv2D(filters=48, kernel_size=[5, 5], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/conv2d_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_Bias')))(inputs)
depthconv1_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_Bias')))(conv1_1)
conv1_2 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_1_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_1_Bias')))(depthconv1_1)
add1_1 = Add()([conv1_1, conv1_2])
relu1_1 = ReLU()(add1_1)

# Bolock_02
depthconv2_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_1_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_1_Bias')))(relu1_1)
conv2_1 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_2_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_2_Bias')))(depthconv2_1)
add2_1 = Add()([relu1_1, conv2_1])
relu2_1 = ReLU()(add2_1)

# Bolock_03
depthconv3_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_2_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_2_Bias')))(relu2_1)
conv3_1 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_3_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_3_Bias')))(depthconv3_1)
add3_1 = Add()([relu2_1, conv3_1])
relu3_1 = ReLU()(add3_1)

# Bolock_04
depthconv4_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_3_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_3_Bias')))(relu3_1)
conv4_1 = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_5_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_5_Bias')))(depthconv4_1)
maxpool4_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu3_1)
conv4_2 = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_4_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_4_Bias')))(maxpool4_1)
add4_1 = Add()([conv4_2, conv4_1])
relu4_1 = ReLU()(add4_1)

# Bolock_05
depthconv5_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_4_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_4_Bias')))(relu4_1)
conv5_1 = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_6_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_6_Bias')))(depthconv5_1)
add5_1 = Add()([relu4_1, conv5_1])
relu5_1 = ReLU()(add5_1)

# Bolock_06
depthconv6_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_5_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_5_Bias')))(relu5_1)
conv6_1 = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_7_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_7_Bias')))(depthconv6_1)
add6_1 = Add()([relu5_1, conv6_1])
relu6_1 = ReLU()(add6_1)

# Bolock_07
depthconv7_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_6_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_6_Bias')))(relu6_1)
conv7_1 = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_8_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_8_Bias')))(depthconv7_1)
add7_1 = Add()([relu6_1, conv7_1])
relu7_1 = ReLU()(add7_1)

# Bolock_08
depthconv8_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_7_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_7_Bias')))(relu7_1)
conv8_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_10_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_10_Bias')))(depthconv8_1)
maxpool8_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu7_1)
conv8_2 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_9_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_9_Bias')))(maxpool8_1)
add8_1 = Add()([conv8_2, conv8_1])
relu8_1 = ReLU()(add8_1)

# Bolock_09
depthconv9_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_8_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_8_Bias')))(relu8_1)
conv9_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_11_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_11_Bias')))(depthconv9_1)
add9_1 = Add()([relu8_1, conv9_1])
relu9_1 = ReLU()(add9_1)

# Bolock_10
depthconv10_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_9_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_9_Bias')))(relu9_1)
conv10_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_12_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_12_Bias')))(depthconv10_1)
add10_1 = Add()([relu9_1, conv10_1])
relu10_1 = ReLU()(add10_1)

# Bolock_11
depthconv11_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_10_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_10_Bias')))(relu10_1)
conv11_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_13_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_13_Bias')))(depthconv11_1)
add11_1 = Add()([relu10_1, conv11_1])
relu11_1 = ReLU()(add11_1)

# Bolock_12
depthconv12_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_11_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_11_Bias')))(relu11_1)
conv12_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_14_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_14_Bias')))(depthconv12_1)
add12_1 = Add()([relu11_1, conv12_1])
relu12_1 = ReLU()(add12_1)

# Bolock_13
depthconv13_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_12_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_12_Bias')))(relu12_1)
conv13_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_15_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_15_Bias')))(depthconv13_1)
add13_1 = Add()([relu12_1, conv13_1])
relu13_1 = ReLU()(add13_1)

# Bolock_14
depthconv14_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_13_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_13_Bias')))(relu13_1)
conv14_1 = Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_16_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_16_Bias')))(depthconv14_1)
add14_1 = Add()([relu13_1, conv14_1])
relu14_1 = ReLU()(add14_1)

# Bolock_15
depthconv15_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_14_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_14_Bias')))(relu14_1)
conv15_1 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_18_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_18_Bias')))(depthconv15_1)
maxpool15_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu14_1)
conv15_2 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_17_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_17_Bias')))(maxpool15_1)
add15_1 = Add()([conv15_2, conv15_1])
relu15_1 = ReLU()(add15_1)

# Bolock_16
depthconv16_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_15_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_15_Bias')))(relu15_1)
conv16_1 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_19_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_19_Bias')))(depthconv16_1)
add16_1 = Add()([relu15_1, conv16_1])
relu16_1 = ReLU()(add16_1)

# Bolock_17
depthconv17_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_16_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_16_Bias')))(relu16_1)
conv17_1 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_20_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_20_Bias')))(depthconv17_1)
add17_1 = Add()([relu16_1, conv17_1])
relu17_1 = ReLU()(add17_1)

# Bolock_18
depthconv18_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_17_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_17_Bias')))(relu17_1)
conv18_1 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_21_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_21_Bias')))(depthconv18_1)
add18_1 = Add()([relu17_1, conv18_1])
relu18_1 = ReLU()(add18_1)

# Bolock_19
depthconv19_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_18_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_18_Bias')))(relu18_1)
conv19_1 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_22_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_22_Bias')))(depthconv19_1)
add19_1 = Add()([relu18_1, conv19_1])
relu19_1 = ReLU()(add19_1)

# Bolock_20
depthconv20_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_19_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_19_Bias')))(relu19_1)
conv20_1 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_23_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_23_Bias')))(depthconv20_1)
add20_1 = Add()([relu19_1, conv20_1])
relu20_1 = ReLU()(add20_1)

# Bolock_21
depthconv21_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_20_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_20_Bias')))(relu20_1)
conv21_1 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_24_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_24_Bias')))(depthconv21_1)
add21_1 = Add()([relu20_1, conv21_1])
relu21_1 = ReLU()(add21_1)

# Bolock_22
depthconv22_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_21_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_21_Bias')))(relu21_1)
conv22_1 = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_25_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_25_Bias')))(depthconv22_1)
add22_1 = Add()([relu21_1, conv22_1])
relu22_1 = ReLU()(add22_1)

# Bolock_23
depthconv23_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[2, 2], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_22_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_22_Bias')))(relu22_1)
conv23_1 = Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_27_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_27_Bias')))(depthconv23_1)
maxpool23_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu22_1)
conv23_2 = Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_26_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_26_Bias')))(maxpool23_1)
add23_1 = Add()([conv23_2, conv23_1])
relu23_1 = ReLU()(add23_1)

# Bolock_24
depthconv24_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_23_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_23_Bias')))(relu23_1)
conv24_1 = Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_28_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_28_Bias')))(depthconv24_1)
add24_1 = Add()([relu23_1, conv24_1])
relu24_1 = ReLU()(add24_1)

# Bolock_25
depthconv25_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_24_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_24_Bias')))(relu24_1)
conv25_1 = Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_29_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_29_Bias')))(depthconv25_1)
add25_1 = Add()([relu24_1, conv25_1])
relu25_1 = ReLU()(add25_1)

# Bolock_26
depthconv26_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_25_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_25_Bias')))(relu25_1)
conv26_1 = Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_30_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_30_Bias')))(depthconv26_1)
add26_1 = Add()([relu25_1, conv26_1])
relu26_1 = ReLU()(add26_1)

# Bolock_27
depthconv27_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_26_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_26_Bias')))(relu26_1)
conv27_1 = Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_31_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_31_Bias')))(depthconv27_1)
add27_1 = Add()([relu26_1, conv27_1])
relu27_1 = ReLU()(add27_1)

# Bolock_28
depthconv28_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_27_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_27_Bias')))(relu27_1)
conv28_1 = Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_32_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_32_Bias')))(depthconv28_1)
add28_1 = Add()([relu27_1, conv28_1])
relu28_1 = ReLU()(add28_1)

# Bolock_29
depthconv29_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/depthwise_conv2d_28_Kernel')), bias_initializer=Constant(np.load('weights/depthwise_conv2d_28_Bias')))(relu28_1)
conv29_1 = Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/conv2d_33_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/conv2d_33_Bias')))(depthconv29_1)
add29_1 = Add()([relu28_1, conv29_1])
relu29_1 = ReLU()(add29_1)

# Bolock_30
conv30_1 = Conv2D(filters=2, kernel_size=[1, 1], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/classificator_8_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/classificator_8_Bias')))(relu22_1)
reshape30_1 = tf.reshape(conv30_1, (1, 512, 1))
conv30_2 = Conv2D(filters=6, kernel_size=[1, 1], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/classificator_16_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/classificator_16_Bias')))(relu29_1)
reshape30_2 = tf.reshape(conv30_2, (1, 384, 1))
concat30_1 = Concatenate(axis=1, name='classificators')([reshape30_1, reshape30_2])

conv30_3 = Conv2D(filters=24, kernel_size=[1, 1], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/regressor_8_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/regressor_8_Bias')))(relu22_1)
reshape30_3 = tf.reshape(conv30_3, (1, 512, 12))
conv30_4 = Conv2D(filters=72, kernel_size=[1, 1], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/regressor_16_Kernel').transpose(1,2,3,0)), bias_initializer=Constant(np.load('weights/regressor_16_Bias')))(relu29_1)
reshape30_4 = tf.reshape(conv30_4, (1, 384, 12))
concat30_2 = Concatenate(axis=1, name='regressors')([reshape30_3, reshape30_4])

model = Model(inputs=inputs, outputs=[concat30_1, concat30_2])

model.summary()

tf.saved_model.save(model, 'saved_model_pose_detection')
model.save('pose_detection.h5')


# No Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('pose_detection_128x128_float32.tflite', 'wb') as w:
    w.write(tflite_model)
print("tflite convert complete! - pose_detection_128x128_float32.tflite")


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
with open('pose_detection_128x128_weight_quant.tflite', 'wb') as w:
    w.write(tflite_model)
print("Weight Quantization complete! - pose_detection_128x128_weight_quant.tflite")


def representative_dataset_gen():
    for image in raw_test_data:
        image = tf.image.resize(image, (128, 128))
        image = image[np.newaxis,:,:,:]
        yield [image]

raw_test_data = np.load('calibration_data_img_person.npy', allow_pickle=True)


# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('pose_detection_128x128_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - pose_detection_128x128_integer_quant.tflite")


# Full Integer Quantization - Input/Output=int8
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('pose_detection_128x128_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - pose_detection_128x128_full_integer_quant.tflite")


# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('pose_detection_128x128_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - pose_detection_128x128_float16_quant.tflite")


# EdgeTPU
import subprocess
result = subprocess.check_output(["edgetpu_compiler", "-s", "pose_detection_128x128_full_integer_quant.tflite"])
print(result)
