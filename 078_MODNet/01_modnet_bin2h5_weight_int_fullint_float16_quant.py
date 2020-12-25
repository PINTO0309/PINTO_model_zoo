### tensorflow==2.3.1

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
### https://www.tensorflow.org/api_docs/python/tf/keras/backend/resize_images
### https://www.tensorflow.org/api_docs/python/tf/keras/activations/tanh

### https://www.tensorflow.org/api_docs/python/tf/math/multiply
### https://www.tensorflow.org/api_docs/python/tf/linalg/matmul
### https://www.tensorflow.org/api_docs/python/tf/math/sigmoid
### https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/examples/digit_classifier/ml/mnist_tflite.ipynb#scrollTo=2fXStjR4mzkR

### How to initialize a convolution layer with an arbitrary kernel in Keras? https://stackoverrun.com/ja/q/12269118

###  saved_model_cli show --dir saved_model_512x512/ --tag_set serve --signature_def serving_default

'''
The given SavedModel SignatureDef contains the following input(s):
  inputs['input'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 512, 512, 3)
      name: serving_default_input:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['tf_op_layer_Sigmoid_1'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 32, 32, 1)
      name: StatefulPartitionedCall:0
  outputs['tf_op_layer_Sigmoid_2'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 512, 512, 1)
      name: StatefulPartitionedCall:1
  outputs['tf_op_layer_Sigmoid_3'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 512, 512, 1)
      name: StatefulPartitionedCall:2
Method name is: tensorflow/serving/predict
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, ReLU, MaxPool2D, Reshape, Concatenate, ZeroPadding2D, Layer, Lambda
from tensorflow.keras.initializers import Constant
from tensorflow.keras.backend import resize_images
from tensorflow.keras.activations import tanh
from tensorflow.math import sigmoid
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import sys

# tmp = np.load('weights_bk/256x256/FP32/131_mean_Fused_Mul_49904992_const.npy').transpose(1,2,3,0)
# # tmp = np.load('weights/{}_{}x{}/FP32/data_add_46134618_copy_const.npy').flatten()#.transpose(0,2,3,1)#.flatten()
# print(tmp.shape)
# print(tmp)
# import sys
# sys.exit(0)

# def init_f(shape, dtype=None):
#        ker = np.load('weights/{}_{}x{}/FP32/depthwise_conv2d_Kernel')
#        print(shape)
#        return ker

# sys.exit(0)

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

height = 512
width  = 512

inputs = Input(shape=(height, width, 3), batch_size=1, name='input')

# Block_01
conv_1_1 = Conv2D(filters=32,
                  kernel_size=[3, 3],
                  strides=[2, 2],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_824_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_0_Dims8668_copy_const.npy').flatten()))(inputs)
relu6_1_1 = tf.nn.relu6(conv_1_1)
depthconv_1_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                depth_multiplier=1,
                                dilation_rate=[1, 1],
                                depthwise_initializer=Constant(np.load('weights/3139831401_const.npy').transpose(3,4,1,2,0)),
                                bias_initializer=Constant(np.load('weights/Conv_2_Dims8896_copy_const.npy').flatten()))(relu6_1_1)
relu6_1_2 = tf.nn.relu6(depthconv_1_1)
conv_1_2 = Conv2D(filters=16,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_830_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_4_Dims8932_copy_const.npy').flatten()))(relu6_1_2)

# Block_02
conv_2_1 = Conv2D(filters=96,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_833_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_5_Dims8878_copy_const.npy').flatten()))(conv_1_2)
relu6_2_1 = tf.nn.relu6(conv_2_1)
depthconv_2_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                strides=[2, 2],
                                padding='same',
                                depth_multiplier=1,
                                dilation_rate=[1, 1],
                                depthwise_initializer=Constant(np.load('weights/3141031413_const.npy').transpose(3,4,1,2,0)),
                                bias_initializer=Constant(np.load('weights/Conv_7_Dims9010_copy_const.npy').flatten()))(relu6_2_1)
relu6_2_2 = tf.nn.relu6(depthconv_2_1)
conv_2_2 = Conv2D(filters=24,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_839_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_9_Dims8860_copy_const.npy').flatten()))(relu6_2_2)

# Block_03
conv_3_1 = Conv2D(filters=144,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_842_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_10_Dims8650_copy_const.npy').flatten()))(conv_2_2)
relu6_3_1 = tf.nn.relu6(conv_3_1)
depthconv_3_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                depth_multiplier=1,
                                dilation_rate=[1, 1],
                                depthwise_initializer=Constant(np.load('weights/3139031393_const.npy').transpose(3,4,1,2,0)),
                                bias_initializer=Constant(np.load('weights/Conv_12_Dims8866_copy_const.npy').flatten()))(relu6_3_1)
relu6_3_2 = tf.nn.relu6(depthconv_3_1)
conv_3_2 = Conv2D(filters=24,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_848_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_14_Dims8998_copy_const.npy').flatten()))(relu6_3_2)
add_3_1 = Add()([conv_3_2, conv_2_2])

# Block_04
conv_4_1 = Conv2D(filters=144,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_851_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_16_Dims8674_copy_const.npy').flatten()))(add_3_1)
relu6_4_1 = tf.nn.relu6(conv_4_1)
depthconv_4_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                strides=[2, 2],
                                padding='same',
                                depth_multiplier=1,
                                dilation_rate=[1, 1],
                                depthwise_initializer=Constant(np.load('weights/3137831381_const.npy').transpose(3,4,1,2,0)),
                                bias_initializer=Constant(np.load('weights/Conv_18_Dims8782_copy_const.npy').flatten()))(relu6_4_1)
relu6_4_2 = tf.nn.relu6(depthconv_4_1)
conv_4_2 = Conv2D(filters=32,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_857_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_20_Dims8848_copy_const.npy').flatten()))(relu6_4_2)

# Block_05
conv_5_1 = Conv2D(filters=192,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_860_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_21_Dims8662_copy_const.npy').flatten()))(conv_4_2)
relu6_5_1 = tf.nn.relu6(conv_5_1)
depthconv_5_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                depth_multiplier=1,
                                dilation_rate=[1, 1],
                                depthwise_initializer=Constant(np.load('weights/3135431357_const.npy').transpose(3,4,1,2,0)),
                                bias_initializer=Constant(np.load('weights/Conv_23_Dims8680_copy_const.npy').flatten()))(relu6_5_1)
relu6_5_2 = tf.nn.relu6(depthconv_5_1)
conv_5_2 = Conv2D(filters=32,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_866_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_25_Dims8950_copy_const.npy').flatten()))(relu6_5_2)
add_5_1 = Add()([conv_5_2, conv_4_2])

# Block_06
conv_6_1 = Conv2D(filters=192,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_869_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_27_Dims8968_copy_const.npy').flatten()))(add_5_1)
relu6_6_1 = tf.nn.relu6(conv_6_1)
depthconv_6_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                depth_multiplier=1,
                                dilation_rate=[1, 1],
                                depthwise_initializer=Constant(np.load('weights/3140631409_const.npy').transpose(3,4,1,2,0)),
                                bias_initializer=Constant(np.load('weights/Conv_29_Dims8914_copy_const.npy').flatten()))(relu6_6_1)
relu6_6_2 = tf.nn.relu6(depthconv_6_1)
conv_6_2 = Conv2D(filters=32,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_875_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_31_Dims8596_copy_const.npy').flatten()))(relu6_6_2)
add_6_1 = Add()([conv_6_2, add_5_1])

# Block_07
conv_7_1 = Conv2D(filters=192,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_878_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_33_Dims8734_copy_const.npy').flatten()))(add_6_1)
relu6_7_1 = tf.nn.relu6(conv_7_1)
depthconv_7_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                strides=[2, 2],
                                padding='same',
                                depth_multiplier=1,
                                dilation_rate=[1, 1],
                                depthwise_initializer=Constant(np.load('weights/3136231365_const.npy').transpose(3,4,1,2,0)),
                                bias_initializer=Constant(np.load('weights/Conv_35_Dims8704_copy_const.npy').flatten()))(relu6_7_1)
relu6_7_2 = tf.nn.relu6(depthconv_7_1)
conv_7_2 = Conv2D(filters=64,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_884_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_37_Dims8920_copy_const.npy').flatten()))(relu6_7_2)

# Block_08
conv_8_1 = Conv2D(filters=384,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_887_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_38_Dims8872_copy_const.npy').flatten()))(conv_7_2)
relu6_8_1 = tf.nn.relu6(conv_8_1)
depthconv_8_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                depth_multiplier=1,
                                dilation_rate=[1, 1],
                                depthwise_initializer=Constant(np.load('weights/3134631349_const.npy').transpose(3,4,1,2,0)),
                                bias_initializer=Constant(np.load('weights/Conv_40_Dims8614_copy_const.npy').flatten()))(relu6_8_1)
relu6_8_2 = tf.nn.relu6(depthconv_8_1)
conv_8_2 = Conv2D(filters=64,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_893_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_42_Dims8986_copy_const.npy').flatten()))(relu6_8_2)
add_8_1 = Add()([conv_8_2, conv_7_2])

# Block_09
conv_9_1 = Conv2D(filters=384,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_896_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_44_Dims8956_copy_const.npy').flatten()))(add_8_1)
relu6_9_1 = tf.nn.relu6(conv_9_1)
depthconv_9_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                depth_multiplier=1,
                                dilation_rate=[1, 1],
                                depthwise_initializer=Constant(np.load('weights/3136631369_const.npy').transpose(3,4,1,2,0)),
                                bias_initializer=Constant(np.load('weights/Conv_46_Dims8716_copy_const.npy').flatten()))(relu6_9_1)
relu6_9_2 = tf.nn.relu6(depthconv_9_1)
conv_9_2 = Conv2D(filters=64,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_902_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_48_Dims8722_copy_const.npy').flatten()))(relu6_9_2)
add_9_1 = Add()([conv_9_2, add_8_1])

# Block_10
conv_10_1 = Conv2D(filters=384,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_905_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_50_Dims9004_copy_const.npy').flatten()))(add_9_1)
relu6_10_1 = tf.nn.relu6(conv_10_1)
depthconv_10_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                depth_multiplier=1,
                                dilation_rate=[1, 1],
                                depthwise_initializer=Constant(np.load('weights/3137031373_const.npy').transpose(3,4,1,2,0)),
                                bias_initializer=Constant(np.load('weights/Conv_52_Dims8728_copy_const.npy').flatten()))(relu6_10_1)
relu6_10_2 = tf.nn.relu6(depthconv_10_1)
conv_10_2 = Conv2D(filters=64,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_911_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_54_Dims8752_copy_const.npy').flatten()))(relu6_10_2)
add_10_1 = Add()([conv_10_2, add_9_1])

# Block_11
conv_11_1 = Conv2D(filters=384,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_914_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_56_Dims8698_copy_const.npy').flatten()))(add_10_1)
relu6_11_1 = tf.nn.relu6(conv_11_1)
depthconv_11_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                depth_multiplier=1,
                                dilation_rate=[1, 1],
                                depthwise_initializer=Constant(np.load('weights/3135031353_const.npy').transpose(3,4,1,2,0)),
                                bias_initializer=Constant(np.load('weights/Conv_58_Dims8644_copy_const.npy').flatten()))(relu6_11_1)
relu6_11_2 = tf.nn.relu6(depthconv_11_1)
conv_11_2 = Conv2D(filters=96,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_920_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_60_Dims8902_copy_const.npy').flatten()))(relu6_11_2)

# Block_12
conv_12_1 = Conv2D(filters=576,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_923_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_61_Dims9016_copy_const.npy').flatten()))(conv_11_2)
relu6_12_1 = tf.nn.relu6(conv_12_1)
depthconv_12_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                depth_multiplier=1,
                                dilation_rate=[1, 1],
                                depthwise_initializer=Constant(np.load('weights/3139431397_const.npy').transpose(3,4,1,2,0)),
                                bias_initializer=Constant(np.load('weights/Conv_63_Dims8884_copy_const.npy').flatten()))(relu6_12_1)
relu6_12_2 = tf.nn.relu6(depthconv_12_1)
conv_12_2 = Conv2D(filters=96,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_929_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_65_Dims8794_copy_const.npy').flatten()))(relu6_12_2)
add_12_1 = Add()([conv_12_2, conv_11_2])

# Block_13
conv_13_1 = Conv2D(filters=576,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_932_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_67_Dims8590_copy_const.npy').flatten()))(add_12_1)
relu6_13_1 = tf.nn.relu6(conv_13_1)
depthconv_13_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                depth_multiplier=1,
                                dilation_rate=[1, 1],
                                depthwise_initializer=Constant(np.load('weights/3140231405_const.npy').transpose(3,4,1,2,0)),
                                bias_initializer=Constant(np.load('weights/Conv_69_Dims8908_copy_const.npy').flatten()))(relu6_13_1)
relu6_13_2 = tf.nn.relu6(depthconv_13_1)
conv_13_2 = Conv2D(filters=96,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_938_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_71_Dims8824_copy_const.npy').flatten()))(relu6_13_2)
add_13_1 = Add()([conv_13_2, add_12_1])

# Block_14
conv_14_1 = Conv2D(filters=576,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_941_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_73_Dims8764_copy_const.npy').flatten()))(add_13_1)
relu6_14_1 = tf.nn.relu6(conv_14_1)
depthconv_14_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                strides=[2, 2],
                                padding='same',
                                depth_multiplier=1,
                                dilation_rate=[1, 1],
                                depthwise_initializer=Constant(np.load('weights/3138231385_const.npy').transpose(3,4,1,2,0)),
                                bias_initializer=Constant(np.load('weights/Conv_75_Dims8836_copy_const.npy').flatten()))(relu6_14_1)
relu6_14_2 = tf.nn.relu6(depthconv_14_1)
conv_14_2 = Conv2D(filters=160,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_947_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_77_Dims8890_copy_const.npy').flatten()))(relu6_14_2)

# Block_15
conv_15_1 = Conv2D(filters=960,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_950_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_78_Dims8608_copy_const.npy').flatten()))(conv_14_2)
relu6_15_1 = tf.nn.relu6(conv_15_1)
depthconv_15_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                depth_multiplier=1,
                                dilation_rate=[1, 1],
                                depthwise_initializer=Constant(np.load('weights/3137431377_const.npy').transpose(3,4,1,2,0)),
                                bias_initializer=Constant(np.load('weights/Conv_80_Dims8776_copy_const.npy').flatten()))(relu6_15_1)
relu6_15_2 = tf.nn.relu6(depthconv_15_1)
conv_15_2 = Conv2D(filters=160,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_956_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_82_Dims8980_copy_const.npy').flatten()))(relu6_15_2)
add_15_1 = Add()([conv_15_2, conv_14_2])

# Block_16
conv_16_1 = Conv2D(filters=960,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_959_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_84_Dims8818_copy_const.npy').flatten()))(add_15_1)
relu6_16_1 = tf.nn.relu6(conv_16_1)
depthconv_16_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                depth_multiplier=1,
                                dilation_rate=[1, 1],
                                depthwise_initializer=Constant(np.load('weights/3138631389_const.npy').transpose(3,4,1,2,0)),
                                bias_initializer=Constant(np.load('weights/Conv_86_Dims8854_copy_const.npy').flatten()))(relu6_16_1)
relu6_16_2 = tf.nn.relu6(depthconv_16_1)
conv_16_2 = Conv2D(filters=160,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_965_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_88_Dims8806_copy_const.npy').flatten()))(relu6_16_2)
add_16_1 = Add()([conv_16_2, add_15_1])

# Block_17
conv_17_1 = Conv2D(filters=960,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_968_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_90_Dims8974_copy_const.npy').flatten()))(add_16_1)
relu6_17_1 = tf.nn.relu6(conv_17_1)
depthconv_17_1 = DepthwiseConv2D(kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                depth_multiplier=1,
                                dilation_rate=[1, 1],
                                depthwise_initializer=Constant(np.load('weights/3135831361_const.npy').transpose(3,4,1,2,0)),
                                bias_initializer=Constant(np.load('weights/Conv_92_Dims8686_copy_const.npy').flatten()))(relu6_17_1)
relu6_17_2 = tf.nn.relu6(depthconv_17_1)
conv_17_2 = Conv2D(filters=320,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_974_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_94_Dims8770_copy_const.npy').flatten()))(relu6_17_2)
conv_17_3 = Conv2D(filters=1280,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  padding='same',
                  dilation_rate=[1, 1],
                  kernel_initializer=Constant(np.load('weights/onnx_initializer_node_977_Output_0_Data__const.npy').transpose(2,3,1,0)),
                  bias_initializer=Constant(np.load('weights/Conv_95_Dims8746_copy_const.npy').flatten()))(conv_17_2)
relu6_17_3 = tf.nn.relu6(conv_17_3)

# Block_18
reducemean_18_1 = tf.math.reduce_mean(relu6_17_3, axis=[1, 2], keepdims=True)
reshape_18_1 = tf.reshape(reducemean_18_1, [1, 1280])
matmul_18_1 = tf.linalg.matmul(reshape_18_1, np.load('weights/MatMul_108_1_port_transpose22003_const.npy'), False, True)
relu_18_1 = tf.nn.relu(matmul_18_1)
matmul_18_2 = tf.linalg.matmul(relu_18_1, np.load('weights/MatMul_110_1_port_transpose21999_const.npy'), False, True)
sigmoid_18_1 = tf.math.sigmoid(matmul_18_2)
reshape_18_2 = tf.reshape(sigmoid_18_1, [1, 1, 1, 1280])
boadcast_18_1 = tf.math.multiply(reshape_18_2, tf.ones([1, 16, 16, 1280]))
multiply_18_1 = tf.math.multiply(boadcast_18_1, relu6_17_3)


# Define Lamda
def upsampling2d_bilinear(x, upsampling_factor_height, upsampling_factor_width):
    h = int(x.shape[1] * upsampling_factor_height)
    w = int(x.shape[2] * upsampling_factor_width)
    return tf.compat.v1.image.resize_bilinear(x, (h, w))

def upsampling2d_nearest(x, upsampling_factor_height, upsampling_factor_width):
    h = int(x.shape[1] * upsampling_factor_height)
    w = int(x.shape[2] * upsampling_factor_width)
    return tf.compat.v1.image.resize_nearest_neighbor(x, (h, w))

# Block_19
upsample_19_1 =  Lambda(upsampling2d_bilinear, arguments={'upsampling_factor_height': 2, 'upsampling_factor_width':  2})(multiply_18_1)
conv_19_1 = Conv2D(filters=96,
                   kernel_size=[5, 5],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_lr_branch.conv_lr16x.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_120_Dims8788_copy_const.npy').flatten()))(upsample_19_1)

# Block_20
split_20_1_first_half, split_20_1_sedond_half = tf.split(conv_19_1, num_or_size_splits=2, axis=-1)

multiply_20_1 = tf.math.multiply(split_20_1_first_half, np.load('weights/data_mul_1472814732_copy_const.npy').transpose(0,2,3,1))
add_20_1 = Add()([multiply_20_1, np.load('weights/data_mul_1472814732_copy_const.npy').transpose(0,2,3,1)])

mean_20_1 = tf.math.reduce_mean(split_20_1_sedond_half, axis=[-1], keepdims=True)
var_20_1 = tf.math.reduce_variance(split_20_1_sedond_half, axis=[-1], keepdims=True)
mvn_20_1 = (split_20_1_sedond_half - mean_20_1) / tf.math.sqrt(var_20_1 + 0.000009999999747378752)
multiply_20_2 = tf.math.multiply(mvn_20_1, np.load('weights/Constant_124_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1))
add_20_2 = Add()([multiply_20_2, np.load('weights/Constant_125_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1)])

concat_20_1 = tf.concat([add_20_1, add_20_2], axis=-1)
relu_20_1 = tf.nn.relu(concat_20_1)
upsample_20_1 =  Lambda(upsampling2d_bilinear, arguments={'upsampling_factor_height': 2, 'upsampling_factor_width':  2})(relu_20_1)
conv_20_1 = Conv2D(filters=32,
                   kernel_size=[5, 5],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_lr_branch.conv_lr8x.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_130_Dims8626_copy_const.npy').flatten()))(upsample_20_1)

# Block_21
split_21_1_first_half, split_21_1_sedond_half = tf.split(conv_20_1, num_or_size_splits=2, axis=-1)

multiply_21_1 = tf.math.multiply(split_21_1_first_half, np.load('weights/data_mul_1473614740_copy_const.npy').transpose(0,2,3,1))
add_21_1 = Add()([multiply_21_1, np.load('weights/data_add_1473714742_copy_const.npy').transpose(0,2,3,1)])

mean_21_1 = tf.math.reduce_mean(split_21_1_sedond_half, axis=[-1], keepdims=True)
var_21_1 = tf.math.reduce_variance(split_21_1_sedond_half, axis=[-1], keepdims=True)
mvn_21_1 = (split_21_1_sedond_half - mean_21_1) / tf.math.sqrt(var_21_1 + 0.000009999999747378752)
multiply_21_2 = tf.math.multiply(mvn_21_1, np.load('weights/Constant_134_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1))
add_21_2 = Add()([multiply_21_2, np.load('weights/Constant_135_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1)])

concat_21_1 = tf.concat([add_21_1, add_21_2], axis=-1)
relu_21_1 = tf.nn.relu(concat_21_1)

# Block_22 - Output.1 ##############################################
conv_22_1 = Conv2D(filters=1,
                   kernel_size=[3, 3],
                   strides=[2, 2],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_lr_branch.conv_lr.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_139_Dims8602_copy_const.npy').flatten()))(relu_21_1)
output_1 = tf.math.sigmoid(conv_22_1)

# Block_23
upsample_23_1 =  Lambda(upsampling2d_bilinear, arguments={'upsampling_factor_height': 2, 'upsampling_factor_width':  2})(relu_21_1)

# Block_24
conv_24_1 = Conv2D(filters=32,
                   kernel_size=[5, 5],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_f_branch.conv_lr4x.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_262_Dims8632_copy_const.npy').flatten()))(upsample_23_1)

split_24_1_first_half, split_24_1_sedond_half = tf.split(conv_24_1, num_or_size_splits=2, axis=-1)

multiply_24_1 = tf.math.multiply(split_24_1_first_half, np.load('weights/data_mul_1480814812_copy_const.npy').transpose(0,2,3,1))
add_24_1 = Add()([multiply_24_1, np.load('weights/data_add_1480914814_copy_const.npy').transpose(0,2,3,1)])

mean_24_1 = tf.math.reduce_mean(split_24_1_sedond_half, axis=[-1], keepdims=True)
var_24_1 = tf.math.reduce_variance(split_24_1_sedond_half, axis=[-1], keepdims=True)
mvn_24_1 = (split_24_1_sedond_half - mean_24_1) / tf.math.sqrt(var_24_1 + 0.000009999999747378752)
multiply_24_2 = tf.math.multiply(mvn_24_1, np.load('weights/Constant_266_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1))
add_24_2 = Add()([multiply_24_2, np.load('weights/Constant_267_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1)])

concat_24_1 = tf.concat([add_24_1, add_24_2], axis=-1)
relu_24_1 = tf.nn.relu(concat_24_1)
upsample_24_2 =  Lambda(upsampling2d_bilinear, arguments={'upsampling_factor_height': 2, 'upsampling_factor_width':  2})(relu_24_1)


# Block_25
conv_25_1 = Conv2D(filters=32,
                   kernel_size=[1, 1],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_hr_branch.tohr_enc2x.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_143_Dims8710_copy_const.npy').flatten()))(conv_1_2)

split_25_1_first_half, split_25_1_sedond_half = tf.split(conv_25_1, num_or_size_splits=2, axis=-1)

multiply_25_1 = tf.math.multiply(split_25_1_first_half, np.load('weights/data_mul_14700_copy_const.npy').transpose(0,2,3,1))
add_25_1 = Add()([multiply_25_1, np.load('weights/data_add_14702_copy_const.npy').transpose(0,2,3,1)])

mean_25_1 = tf.math.reduce_mean(split_25_1_sedond_half, axis=[-1], keepdims=True)
var_25_1 = tf.math.reduce_variance(split_25_1_sedond_half, axis=[-1], keepdims=True)
mvn_25_1 = (split_25_1_sedond_half - mean_25_1) / tf.math.sqrt(var_25_1 + 0.000009999999747378752)
multiply_25_2 = tf.math.multiply(mvn_25_1, np.load('weights/Constant_147_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1))
add_25_2 = Add()([multiply_25_2, np.load('weights/Constant_148_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1)])

concat_25_1 = tf.concat([add_25_1, add_25_2], axis=-1)
relu_25_1 = tf.nn.relu(concat_25_1)

# Block_26
resize_26_1 = Lambda(upsampling2d_bilinear, arguments={'upsampling_factor_height': 0.5, 'upsampling_factor_width':  0.5})(inputs)
concat_26_1 = tf.concat([relu_25_1, resize_26_1], axis=-1)
conv_26_1 = Conv2D(filters=32,
                   kernel_size=[3, 3],
                   strides=[2, 2],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_hr_branch.conv_enc2x.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_153_Dims8800_copy_const.npy').flatten()))(concat_26_1)

split_26_1_first_half, split_26_1_sedond_half = tf.split(conv_26_1, num_or_size_splits=2, axis=-1)

multiply_26_1 = tf.math.multiply(split_26_1_first_half, np.load('weights/data_mul_1470414708_copy_const.npy').transpose(0,2,3,1))
add_26_1 = Add()([multiply_26_1, np.load('weights/data_add_1470514710_copy_const.npy').transpose(0,2,3,1)])

mean_26_1 = tf.math.reduce_mean(split_26_1_sedond_half, axis=[-1], keepdims=True)
var_26_1 = tf.math.reduce_variance(split_26_1_sedond_half, axis=[-1], keepdims=True)
mvn_26_1 = (split_26_1_sedond_half - mean_26_1) / tf.math.sqrt(var_26_1 + 0.000009999999747378752)
multiply_26_2 = tf.math.multiply(mvn_26_1, np.load('weights/Constant_157_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1))
add_26_2 = Add()([multiply_26_2, np.load('weights/Constant_158_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1)])

concat_26_1 = tf.concat([add_26_1, add_26_2], axis=-1)
relu_26_1 = tf.nn.relu(concat_26_1)

# Block_27
conv_27_1 = Conv2D(filters=32,
                   kernel_size=[1, 1],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_hr_branch.tohr_enc4x.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_162_Dims8656_copy_const.npy').flatten()))(add_3_1)

split_27_1_first_half, split_27_1_sedond_half = tf.split(conv_27_1, num_or_size_splits=2, axis=-1)

multiply_27_1 = tf.math.multiply(split_27_1_first_half, np.load('weights/data_mul_1471214716_copy_const.npy').transpose(0,2,3,1))
add_27_1 = Add()([multiply_27_1, np.load('weights/data_add_1471314718_copy_const.npy').transpose(0,2,3,1)])

mean_27_1 = tf.math.reduce_mean(split_27_1_sedond_half, axis=[-1], keepdims=True)
var_27_1 = tf.math.reduce_variance(split_27_1_sedond_half, axis=[-1], keepdims=True)
mvn_27_1 = (split_27_1_sedond_half - mean_27_1) / tf.math.sqrt(var_27_1 + 0.000009999999747378752)
multiply_27_2 = tf.math.multiply(mvn_27_1, np.load('weights/Constant_166_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1))
add_27_2 = Add()([multiply_27_2, np.load('weights/Constant_167_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1)])

concat_27_1 = tf.concat([add_27_1, add_27_2], axis=-1)
relu_27_1 = tf.nn.relu(concat_27_1)

concat_27_2 = tf.concat([relu_27_1, relu_26_1], axis=-1)

# Block_28
conv_28_1 = Conv2D(filters=64,
                   kernel_size=[3, 3],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_hr_branch.conv_enc4x.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_172_Dims8812_copy_const.npy').flatten()))(concat_27_2)

split_28_1_first_half, split_28_1_sedond_half = tf.split(conv_28_1, num_or_size_splits=2, axis=-1)

multiply_28_1 = tf.math.multiply(split_28_1_first_half, np.load('weights/data_mul_1472014724_copy_const.npy').transpose(0,2,3,1))
add_28_1 = Add()([multiply_28_1, np.load('weights/data_add_1472114726_copy_const.npy').transpose(0,2,3,1)])

mean_28_1 = tf.math.reduce_mean(split_28_1_sedond_half, axis=[-1], keepdims=True)
var_28_1 = tf.math.reduce_variance(split_28_1_sedond_half, axis=[-1], keepdims=True)
mvn_28_1 = (split_28_1_sedond_half - mean_28_1) / tf.math.sqrt(var_28_1 + 0.000009999999747378752)
multiply_28_2 = tf.math.multiply(mvn_28_1, np.load('weights/Constant_176_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1))
add_28_2 = Add()([multiply_28_2, np.load('weights/Constant_177_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1)])

concat_28_1 = tf.concat([add_28_1, add_28_2], axis=-1)
relu_28_1 = tf.nn.relu(concat_28_1)

# Bloack_29
resize_29_1 = Lambda(upsampling2d_bilinear, arguments={'upsampling_factor_height': 0.25, 'upsampling_factor_width':  0.25})(inputs)

# Block_30 - Concat_182
concat_30_1 = tf.concat([upsample_23_1, relu_28_1, resize_29_1], axis=-1)

# Block_31
conv_31_1 = Conv2D(filters=64,
                   kernel_size=[3, 3],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_hr_branch.conv_hr4x.0.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_183_Dims8938_copy_const.npy').flatten()))(concat_30_1)

split_31_1_first_half, split_31_1_sedond_half = tf.split(conv_31_1, num_or_size_splits=2, axis=-1)

multiply_31_1 = tf.math.multiply(split_31_1_first_half, np.load('weights/data_mul_1474414748_copy_const.npy').transpose(0,2,3,1))
add_31_1 = Add()([multiply_31_1, np.load('weights/data_add_1474514750_copy_const.npy').transpose(0,2,3,1)])

mean_31_1 = tf.math.reduce_mean(split_31_1_sedond_half, axis=[-1], keepdims=True)
var1 = tf.math.reduce_variance(split_31_1_sedond_half, axis=[-1], keepdims=True)
mvn_31_1 = (split_31_1_sedond_half - mean_31_1) / tf.math.sqrt(var1 + 0.000009999999747378752)
multiply_31_2 = tf.math.multiply(mvn_31_1, np.load('weights/Constant_187_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1))
add_31_2 = Add()([multiply_31_2, np.load('weights/Constant_188_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1)])

concat_31_1 = tf.concat([add_31_1, add_31_2], axis=-1)
relu_31_1 = tf.nn.relu(concat_31_1)

conv_31_2 = Conv2D(filters=64,
                   kernel_size=[3, 3],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_hr_branch.conv_hr4x.1.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_192_Dims8962_copy_const.npy').flatten()))(relu_31_1)

split_31_2_first_half, split_31_2_sedond_half = tf.split(conv_31_2, num_or_size_splits=2, axis=-1)

multiply_31_3 = tf.math.multiply(split_31_2_first_half, np.load('weights/data_mul_1474414748_copy_const.npy').transpose(0,2,3,1))
add_31_3 = Add()([multiply_31_3, np.load('weights/data_add_1474514750_copy_const.npy').transpose(0,2,3,1)])

mean_31_2 = tf.math.reduce_mean(split_31_2_sedond_half, axis=[-1], keepdims=True)
var_31_2 = tf.math.reduce_variance(split_31_2_sedond_half, axis=[-1], keepdims=True)
mvn_31_2 = (split_31_2_sedond_half - mean_31_2) / tf.math.sqrt(var_31_2 + 0.000009999999747378752)
multiply_31_4 = tf.math.multiply(mvn_31_2, np.load('weights/Constant_187_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1))
add_31_4 = Add()([multiply_31_4, np.load('weights/Constant_188_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1)])

concat_31_2 = tf.concat([add_31_3, add_31_4], axis=-1)
relu_31_2 = tf.nn.relu(concat_31_2)

# Block_32
conv_32_1 = Conv2D(filters=32,
                   kernel_size=[3, 3],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_hr_branch.conv_hr4x.2.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_201_Dims8638_copy_const.npy').flatten()))(relu_31_2)

split_32_1_first_half, split_32_1_sedond_half = tf.split(conv_32_1, num_or_size_splits=2, axis=-1)

multiply_32_1 = tf.math.multiply(split_32_1_first_half, np.load('weights/data_mul_1476014764_copy_const.npy').transpose(0,2,3,1))
add_32_1 = Add()([multiply_32_1, np.load('weights/data_add_1476114766_copy_const.npy').transpose(0,2,3,1)])

mean_32_1 = tf.math.reduce_mean(split_32_1_sedond_half, axis=[-1], keepdims=True)
var_32_1 = tf.math.reduce_variance(split_32_1_sedond_half, axis=[-1], keepdims=True)
mvn_32_1 = (split_32_1_sedond_half - mean_32_1) / tf.math.sqrt(var_32_1 + 0.000009999999747378752)
multiply_32_2 = tf.math.multiply(mvn_32_1, np.load('weights/Constant_205_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1))
add_32_2 = Add()([multiply_32_2, np.load('weights/Constant_206_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1)])

concat_32_1 = tf.concat([add_32_1, add_32_2], axis=-1)
relu_32_1 = tf.nn.relu(concat_32_1)

upsample_32_1 =  Lambda(upsampling2d_bilinear, arguments={'upsampling_factor_height': 2, 'upsampling_factor_width':  2})(relu_32_1)
concat_32_2 = tf.concat([upsample_32_1, relu_25_1], axis=-1)

# Block_33
conv_33_1 = Conv2D(filters=64,
                   kernel_size=[3, 3],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_hr_branch.conv_hr2x.0.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_212_Dims8620_copy_const.npy').flatten()))(concat_32_2)

split_33_1_first_half, split_33_1_sedond_half = tf.split(conv_33_1, num_or_size_splits=2, axis=-1)

multiply_33_1 = tf.math.multiply(split_33_1_first_half, np.load('weights/data_mul_1476814772_copy_const.npy').transpose(0,2,3,1))
add_33_1 = Add()([multiply_33_1, np.load('weights/data_add_1476914774_copy_const.npy').transpose(0,2,3,1)])

mean_33_1 = tf.math.reduce_mean(split_33_1_sedond_half, axis=[-1], keepdims=True)
var_33_1 = tf.math.reduce_variance(split_33_1_sedond_half, axis=[-1], keepdims=True)
mvn_33_1 = (split_33_1_sedond_half - mean_33_1) / tf.math.sqrt(var_33_1 + 0.000009999999747378752)
multiply_33_2 = tf.math.multiply(mvn_33_1, np.load('weights/Constant_216_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1))
add_33_2 = Add()([multiply_33_2, np.load('weights/Constant_217_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1)])

concat_33_1 = tf.concat([add_33_1, add_33_2], axis=-1)
relu_33_1 = tf.nn.relu(concat_33_1)

# Block_34
conv_34_1 = Conv2D(filters=32,
                   kernel_size=[3, 3],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_hr_branch.conv_hr2x.1.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_221_Dims8740_copy_const.npy').flatten()))(relu_33_1)

split_34_1_first_half, split_34_1_sedond_half = tf.split(conv_34_1, num_or_size_splits=2, axis=-1)

multiply_34_1 = tf.math.multiply(split_34_1_first_half, np.load('weights/data_mul_1477614780_copy_const.npy').transpose(0,2,3,1))
add_34_1 = Add()([multiply_34_1, np.load('weights/data_add_1477714782_copy_const.npy').transpose(0,2,3,1)])

mean_34_1 = tf.math.reduce_mean(split_34_1_sedond_half, axis=[-1], keepdims=True)
var_34_1 = tf.math.reduce_variance(split_34_1_sedond_half, axis=[-1], keepdims=True)
mvn_34_1 = (split_34_1_sedond_half - mean_34_1) / tf.math.sqrt(var_34_1 + 0.000009999999747378752)
multiply_34_2 = tf.math.multiply(mvn_34_1, np.load('weights/Constant_225_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1))
add_34_2 = Add()([multiply_34_2, np.load('weights/Constant_226_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1)])

concat_34_1 = tf.concat([add_34_1, add_34_2], axis=-1)
relu_34_1 = tf.nn.relu(concat_34_1)

# Block_35
conv_35_1 = Conv2D(filters=32,
                   kernel_size=[3, 3],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_hr_branch.conv_hr2x.2.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_230_Dims8992_copy_const.npy').flatten()))(relu_34_1)

split_35_1_first_half, split_35_1_sedond_half = tf.split(conv_35_1, num_or_size_splits=2, axis=-1)

multiply_35_1 = tf.math.multiply(split_35_1_first_half, np.load('weights/data_mul_1478414788_copy_const.npy').transpose(0,2,3,1))
add_35_1 = Add()([multiply_35_1, np.load('weights/data_add_1478514790_copy_const.npy').transpose(0,2,3,1)])

mean_35_1 = tf.math.reduce_mean(split_35_1_sedond_half, axis=[-1], keepdims=True)
var_35_1 = tf.math.reduce_variance(split_35_1_sedond_half, axis=[-1], keepdims=True)
mvn_35_1 = (split_35_1_sedond_half - mean_35_1) / tf.math.sqrt(var_35_1 + 0.000009999999747378752)
multiply_35_2 = tf.math.multiply(mvn_35_1, np.load('weights/Constant_234_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1))
add_35_2 = Add()([multiply_35_2, np.load('weights/Constant_235_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1)])

concat_35_1 = tf.concat([add_35_1, add_35_2], axis=-1)
relu_35_1 = tf.nn.relu(concat_35_1)

# Block_36
conv_36_1 = Conv2D(filters=32,
                   kernel_size=[3, 3],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_hr_branch.conv_hr2x.3.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_239_Dims8692_copy_const.npy').flatten()))(relu_35_1)

split_36_1_first_half, split_36_1_sedond_half = tf.split(conv_36_1, num_or_size_splits=2, axis=-1)

multiply_36_1 = tf.math.multiply(split_36_1_first_half, np.load('weights/data_mul_1479214796_copy_const.npy').transpose(0,2,3,1))
add_36_1 = Add()([multiply_36_1, np.load('weights/data_add_1479314798_copy_const.npy').transpose(0,2,3,1)])

mean_36_1 = tf.math.reduce_mean(split_36_1_sedond_half, axis=[-1], keepdims=True)
var_36_1 = tf.math.reduce_variance(split_36_1_sedond_half, axis=[-1], keepdims=True)
mvn_36_1 = (split_36_1_sedond_half - mean_36_1) / tf.math.sqrt(var_36_1 + 0.000009999999747378752)
multiply_36_2 = tf.math.multiply(mvn_36_1, np.load('weights/Constant_243_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1))
add_36_2 = Add()([multiply_36_2, np.load('weights/Constant_244_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1)])

concat_36_1 = tf.concat([add_36_1, add_36_2], axis=-1)
relu_36_1 = tf.nn.relu(concat_36_1)

# Block_37 - Concat_272
concat_37_1 = tf.concat([upsample_24_2, relu_36_1], axis=-1)

# Block_38 - Concat_249
upsample_38_1 =  Lambda(upsampling2d_bilinear, arguments={'upsampling_factor_height': 2, 'upsampling_factor_width':  2})(relu_36_1)
concat_38_1 = tf.concat([upsample_38_1, inputs], axis=-1)

# Block_39 - Conv_250/WithoutBiases
conv_39_1 = Conv2D(filters=32,
                   kernel_size=[3, 3],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_hr_branch.conv_hr.0.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_250_Dims8842_copy_const.npy').flatten()))(concat_38_1)

split_39_1_first_half, split_39_1_sedond_half = tf.split(conv_39_1, num_or_size_splits=2, axis=-1)

multiply_39_1 = tf.math.multiply(split_39_1_first_half, np.load('weights/data_mul_1480014804_copy_const.npy').transpose(0,2,3,1))
add_39_1 = Add()([multiply_39_1, np.load('weights/data_add_1480114806_copy_const.npy').transpose(0,2,3,1)])

mean_39_1 = tf.math.reduce_mean(split_39_1_sedond_half, axis=[-1], keepdims=True)
var_39_1 = tf.math.reduce_variance(split_39_1_sedond_half, axis=[-1], keepdims=True)
mvn_39_1 = (split_39_1_sedond_half - mean_39_1) / tf.math.sqrt(var_39_1 + 0.000009999999747378752)
multiply_39_2 = tf.math.multiply(mvn_39_1, np.load('weights/Constant_254_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1))
add_39_2 = Add()([multiply_39_2, np.load('weights/Constant_255_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1)])

concat_39_1 = tf.concat([add_39_1, add_39_2], axis=-1)
relu_39_1 = tf.nn.relu(concat_39_1)

# Block_40 -  Output.2 ##############################################
conv_40_1 = Conv2D(filters=1,
                   kernel_size=[1, 1],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_hr_branch.conv_hr.1.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_259_Dims8944_copy_const.npy').flatten()))(relu_39_1)
output_2 = tf.math.sigmoid(conv_40_1)

# Block_41 - Conv_273/WithoutBiases
conv_41_1 = Conv2D(filters=32,
                   kernel_size=[3, 3],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_f_branch.conv_f2x.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_273_Dims8926_copy_const.npy').flatten()))(concat_37_1)

split_41_1_first_half, split_41_1_sedond_half = tf.split(conv_41_1, num_or_size_splits=2, axis=-1)

multiply_41_1 = tf.math.multiply(split_41_1_first_half, np.load('weights/data_mul_1481614820_copy_const.npy').transpose(0,2,3,1))
add_41_1 = Add()([multiply_41_1, np.load('weights/data_add_1481714822_copy_const.npy').transpose(0,2,3,1)])

mean_41_1 = tf.math.reduce_mean(split_41_1_sedond_half, axis=[-1], keepdims=True)
var_41_1 = tf.math.reduce_variance(split_41_1_sedond_half, axis=[-1], keepdims=True)
mvn_41_1 = (split_41_1_sedond_half - mean_41_1) / tf.math.sqrt(var_41_1 + 0.000009999999747378752)
multiply_41_2 = tf.math.multiply(mvn_41_1, np.load('weights/Constant_277_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1))
add_41_2 = Add()([multiply_41_2, np.load('weights/Constant_278_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1)])

concat_41_1 = tf.concat([add_41_1, add_41_2], axis=-1)
relu_41_1 = tf.nn.relu(concat_41_1)

upsample_41_1 =  Lambda(upsampling2d_bilinear, arguments={'upsampling_factor_height': 2, 'upsampling_factor_width':  2})(relu_41_1)
concat_41_2 = tf.concat([upsample_41_1, inputs], axis=-1) # Concat_283

# Block_42 - Conv_284/WithoutBiases
conv_42_1 = Conv2D(filters=16,
                   kernel_size=[3, 3],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_f_branch.conv_f.0.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_284_Dims8830_copy_const.npy').flatten()))(concat_41_2)

split_42_1_first_half, split_42_1_sedond_half = tf.split(conv_42_1, num_or_size_splits=2, axis=-1)

multiply_42_1 = tf.math.multiply(split_42_1_first_half, np.load('weights/data_mul_1482414828_copy_const.npy').transpose(0,2,3,1))
add_42_1 = Add()([multiply_42_1, np.load('weights/data_add_1482514830_copy_const.npy').transpose(0,2,3,1)])

mean_42_1 = tf.math.reduce_mean(split_42_1_sedond_half, axis=[-1], keepdims=True)
var_42_1 = tf.math.reduce_variance(split_42_1_sedond_half, axis=[-1], keepdims=True)
mvn_42_1 = (split_42_1_sedond_half - mean_42_1) / tf.math.sqrt(var_42_1 + 0.000009999999747378752)
multiply_42_2 = tf.math.multiply(mvn_42_1, np.load('weights/Constant_288_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1))
add_42_2 = Add()([multiply_42_2, np.load('weights/Constant_289_Output_0_Data__copy_copy_const.npy').transpose(0,2,3,1)])

concat_42_1 = tf.concat([add_42_1, add_42_2], axis=-1)
relu_42_1 = tf.nn.relu(concat_42_1)

# Block_43 - Output.3 ##############################################
conv_43_1 = Conv2D(filters=1,
                   kernel_size=[1, 1],
                   strides=[1, 1],
                   padding='same',
                   dilation_rate=[1, 1],
                   kernel_initializer=Constant(np.load('weights/onnx_initializer_node_f_branch.conv_f.1.layers.0.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                   bias_initializer=Constant(np.load('weights/Conv_293_Dims8758_copy_const.npy').flatten()))(relu_42_1)
output_3 = tf.math.sigmoid(conv_43_1)



model = Model(inputs=inputs, outputs=[output_1, output_2, output_3])
model.summary()

saved_model_dir = f'saved_model_{height}x{width}'

# Output saved_model
try:
    tf.saved_model.save(model, saved_model_dir)
except Exception as e:
    print(f'{Color.RED}ERROR:{Color.RESET}', e)
    import traceback
    traceback.print_exc()

# Output .pb
try:
    full_model = tf.function(lambda inputs: model(inputs))
    full_model = full_model.get_concrete_function(inputs=[tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)])
    frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir='.',
                        name=f'{saved_model_dir}/modnet_{height}x{width}_float32.pb',
                        as_text=False)
except Exception as e:
    print(f'{Color.RED}ERROR:{Color.RESET}', e)
    import traceback
    traceback.print_exc()

# No Quantization - Input/Output=float32
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    with open(f'{saved_model_dir}/modnet_{height}x{width}_float32.tflite', 'wb') as w:
        w.write(tflite_model)
    print(f'tflite convert complete! - {saved_model_dir}/modnet_{height}x{width}_float32.tflite')
except Exception as e:
    print(f'{Color.RED}ERROR:{Color.RESET}', e)
    import traceback
    traceback.print_exc()

# Weight Quantization - Input/Output=float32
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    with open(f'{saved_model_dir}/modnet_{height}x{width}_weight_quant.tflite', 'wb') as w:
        w.write(tflite_model)
    print(f'Weight Quantization complete! - {saved_model_dir}/modnet_{height}x{width}_weight_quant.tflite')
except Exception as e:
    print(f'{Color.RED}ERROR:{Color.RESET}', e)
    import traceback
    traceback.print_exc()

# Float16 Quantization - Input/Output=float32
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_quant_model = converter.convert()
    with open(f'{saved_model_dir}/modnet_{height}x{width}_float16_quant.tflite', 'wb') as w:
        w.write(tflite_quant_model)
    print(f'Float16 Quantization complete! - {saved_model_dir}/modnet_{height}x{width}_float16_quant.tflite')
except Exception as e:
    print(f'{Color.RED}ERROR:{Color.RESET}', e)
    import traceback
    traceback.print_exc()

# def representative_dataset_gen():
#     for data in raw_test_data.take(10):
#         image = data['image'].numpy()
#         image = tf.image.resize(image, (height, width))
#         image = image[np.newaxis,:,:,:]
#         image = image / 255
#         yield [image]

# raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)

# # Integer Quantization - Input/Output=float32 - tf-nightly
# try:
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
#     converter.representative_dataset = representative_dataset_gen
#     tflite_quant_model = converter.convert()
#     with open(f'{saved_model_dir}/modnet_{height}x{width}_integer_quant.tflite', 'wb') as w:
#         w.write(tflite_quant_model)
#     print(f'Integer Quantization complete! - {saved_model_dir}/modnet_{height}x{width}_integer_quant.tflite')
# except Exception as e:
#     print(f'{Color.RED}ERROR:{Color.RESET}', e)
#     import traceback
#     traceback.print_exc()

# # Full Integer Quantization - Input/Output=int8 - tf-nightly
# try:
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
#     converter.inference_input_type = tf.int8
#     converter.inference_output_type = tf.int8
#     converter.representative_dataset = representative_dataset_gen
#     tflite_quant_model = converter.convert()
#     with open(f'{saved_model_dir}/modnet_{height}x{width}_full_integer_quant.tflite', 'wb') as w:
#         w.write(tflite_quant_model)
#     print(f'Full Integer Quantization complete! - {saved_model_dir}/modnet_{height}x{width}_full_integer_quant.tflite')
# except Exception as e:
#     print(f'{Color.RED}ERROR:{Color.RESET}', e)
#     import traceback
#     traceback.print_exc()

# # EdgeTPU - tf-nightly
# try:
#     import subprocess
#     result = subprocess.check_output(["edgetpu_compiler", "-s", f"{saved_model_dir}/modnet_{height}x{width}_full_integer_quant.tflite"])
#     print(result)
# except subprocess.CalledProcessError as e:
#     print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
#     import traceback
#     traceback.print_exc()
#     print("-" * 80)
#     print('Please install edgetpu_compiler according to the following website.')
#     print('https://coral.ai/docs/edgetpu/compiler/#system-requirements')

# TensorFlow.js convert
import subprocess
try:
    print(f'{Color.REVERCE}TensorFlow.js Float32 convertion started{Color.RESET}', '=' * 44)
    result = subprocess.check_output(['tensorflowjs_converter',
                                    '--input_format', 'tf_saved_model',
                                    '--output_format', 'tfjs_graph_model',
                                    '--signature_name', 'serving_default',
                                    '--saved_model_tags', 'serve',
                                    saved_model_dir, f'{saved_model_dir}/tfjs_model_float32'],
                                    stderr=subprocess.PIPE).decode('utf-8')
    print(result)
    print(f'{Color.GREEN}TensorFlow.js convertion complete!{Color.RESET} - {saved_model_dir}/tfjs_model_float32')
except subprocess.CalledProcessError as e:
    print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
    import traceback
    traceback.print_exc()
try:
    print(f'{Color.REVERCE}TensorFlow.js Float16 convertion started{Color.RESET}', '=' * 44)
    result = subprocess.check_output(['tensorflowjs_converter',
                                    '--quantize_float16',
                                    '--input_format', 'tf_saved_model',
                                    '--output_format', 'tfjs_graph_model',
                                    '--signature_name', 'serving_default',
                                    '--saved_model_tags', 'serve',
                                    saved_model_dir, f'{saved_model_dir}/tfjs_model_float16'],
                                    stderr=subprocess.PIPE).decode('utf-8')
    print(result)
    print(f'{Color.GREEN}TensorFlow.js convertion complete!{Color.RESET} - {saved_model_dir}/tfjs_model_float16')
except subprocess.CalledProcessError as e:
    print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
    import traceback
    traceback.print_exc()

# TF-TRT (TensorRT) convert
try:
    def input_fn():
        input_shapes = []
        for tf_input in model.inputs:
            input_shapes.append(np.zeros(tf_input.shape).astype(np.float32))
        yield input_shapes

    print(f'{Color.REVERCE}TF-TRT (TensorRT) Float32 convertion started{Color.RESET}', '=' * 40)
    params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP32', maximum_cached_engines=10000)
    converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=saved_model_dir, conversion_params=params)
    converter.convert()
    converter.build(input_fn=input_fn)
    converter.save(f'{saved_model_dir}/tensorrt_saved_model_float32')
    print(f'{Color.GREEN}TF-TRT (TensorRT) convertion complete!{Color.RESET} - {saved_model_dir}/tensorrt_saved_model_float32')
    print(f'{Color.REVERCE}TF-TRT (TensorRT) Float16 convertion started{Color.RESET}', '=' * 40)
    params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP16', maximum_cached_engines=10000)
    converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=saved_model_dir, conversion_params=params)
    converter.convert()
    converter.build(input_fn=input_fn)
    converter.save(f'{saved_model_dir}/tensorrt_saved_model_float16')
    print(f'{Color.GREEN}TF-TRT (TensorRT) convertion complete!{Color.RESET} - {saved_model_dir}/tensorrt_saved_model_float16')
except Exception as e:
    print(f'{Color.RED}ERROR:{Color.RESET}', e)
    import traceback
    traceback.print_exc()
    print(f'{Color.RED}The binary versions of TensorFlow and TensorRT may not be compatible. Please check the version compatibility of each package.{Color.RESET}')

# CoreML convert
try:
    import coremltools as ct
    print(f'{Color.REVERCE}CoreML convertion started{Color.RESET}', '=' * 59)
    mlmodel = ct.convert(saved_model_dir, source='tensorflow')
    mlmodel.save(f'{saved_model_dir}/model_coreml_float32.mlmodel')
    print(f'{Color.GREEN}CoreML convertion complete!{Color.RESET} - {saved_model_dir}/model_coreml_float32.mlmodel')
except Exception as e:
    print(f'{Color.RED}ERROR:{Color.RESET}', e)
    import traceback
    traceback.print_exc()