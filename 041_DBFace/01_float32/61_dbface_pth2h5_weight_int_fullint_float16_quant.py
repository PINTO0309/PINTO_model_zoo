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
### https://www.tensorflow.org/api_docs/python/tf/keras/backend/mean
### https://www.tensorflow.org/api_docs/python/tf/keras/backend/expand_dims
### https://www.tensorflow.org/api_docs/python/tf/shape
### https://www.tensorflow.org/api_docs/python/tf/strided_slice
### https://www.tensorflow.org/api_docs/python/tf/image/resize
### https://www.tensorflow.org/api_docs/python/tf/image/ResizeMethod


### How to initialize a convolution layer with an arbitrary kernel in Keras? https://stackoverrun.com/ja/q/12269118

###  saved_model_cli show --dir saved_model/ --tag_set serve --signature_def serving_default

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, ReLU, MaxPool2D, Reshape, Concatenate, Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.backend import mean, expand_dims, exp
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import sys

# pd = np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :]
# print('pd:', pd.shape)
# print('pd:', pd)

inputs = Input(shape=(512, 512, 3), batch_size=1, name='input')

# Block_01
pad1_1 = tf.pad(inputs, paddings=np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :])
conv1_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[2, 2], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/model_475_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_476_FusedBatchNormV3_nhwc;model_484_Conv2D_nhwc;model_475_Conv2D_nhwc')))(pad1_1)

# Block_02
conv2_1 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/model_478_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_479_FusedBatchNormV3_nhwc;model_484_Conv2D_nhwc;model_478_Conv2D_nhwc')))(conv1_1)
pad2_1 = tf.pad(conv2_1, paddings=np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :])
depthconv2_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="valid", depth_multiplier=1, dilation_rate=[1, 1], activation='relu',
                 depthwise_initializer=Constant(np.load('weights/model_482_FusedBatchNormV3_nhwc;model_481_depthwise_nhwc;model_484_Conv2D_nhwc')),
                 bias_initializer=Constant(np.load('weights/model_482_FusedBatchNormV3_nhwc')))(pad2_1)
conv2_2 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_484_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_485_FusedBatchNormV3_nhwc;model_484_Conv2D_nhwc')))(depthconv2_1)
add2_1 = Add()([conv1_1, conv2_2])

# Block_03
conv3_1 = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/model_487_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_488_FusedBatchNormV3_nhwc;model_490_depthwise_nhwc;model_487_Conv2D_nhwc')))(add2_1)
pad3_1 = tf.pad(conv3_1, paddings=np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :])
depthconv3_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding="valid", depth_multiplier=1, dilation_rate=[1, 1], activation='relu',
                 depthwise_initializer=Constant(np.load('weights/model_491_FusedBatchNormV3_nhwc;model_490_depthwise_nhwc')),
                 bias_initializer=Constant(np.load('weights/model_491_FusedBatchNormV3_nhwc')))(pad3_1)
conv3_2 = Conv2D(filters=24, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_493_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_494_FusedBatchNormV3_nhwc;model_963_Conv2D_nhwc;model_493_Conv2D_nhwc')))(depthconv3_1)

# Block_04
conv4_1 = Conv2D(filters=72, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/model_495_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_496_FusedBatchNormV3_nhwc;model_507_depthwise_nhwc;model_495_Conv2D_nhwc')))(conv3_2)
pad4_1 = tf.pad(conv4_1, paddings=np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :])
depthconv4_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="valid", depth_multiplier=1, dilation_rate=[1, 1], activation='relu',
                 depthwise_initializer=Constant(np.load('weights/model_499_FusedBatchNormV3_nhwc;model_498_depthwise_nhwc;model_507_depthwise_nhwc')),
                 bias_initializer=Constant(np.load('weights/model_499_FusedBatchNormV3_nhwc')))(pad4_1)
conv4_2 = Conv2D(filters=24, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_501_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_502_FusedBatchNormV3_nhwc;model_963_Conv2D_nhwc;model_501_Conv2D_nhwc')))(depthconv4_1)
add4_1 = Add()([conv3_2, conv4_2])

# Block_05
conv5_1 = Conv2D(filters=72, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/model_504_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_505_FusedBatchNormV3_nhwc;model_507_depthwise_nhwc;model_504_Conv2D_nhwc')))(add4_1)
pad5_1 = tf.pad(conv5_1, paddings=np.load('weights/model_776_pad_Pad_paddings')[[0,2,3,1], :])
depthconv5_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[2, 2], padding="valid", depth_multiplier=1, dilation_rate=[1, 1], activation='relu',
                 depthwise_initializer=Constant(np.load('weights/model_508_FusedBatchNormV3_nhwc;model_507_depthwise_nhwc')),
                 bias_initializer=Constant(np.load('weights/model_508_FusedBatchNormV3_nhwc')))(pad5_1)
conv5_2 = Conv2D(filters=40, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_510_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_511_FusedBatchNormV3_nhwc;model_787_Conv2D_nhwc;model_510_Conv2D_nhwc')))(depthconv5_1)

# Block_06
mean6_1 = mean(conv5_2, axis=[1, 2], keepdims=False)
expand6_1 = expand_dims(mean6_1, axis=1)
expand6_2 = expand_dims(expand6_1, axis=1)
conv6_1 = Conv2D(filters=10, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/model_513_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_514_FusedBatchNormV3_nhwc;model_landmark_Conv2D_nhwc;model_513_Conv2D_nhwc')))(expand6_2)
conv6_2 = Conv2D(filters=40, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_516_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_517_FusedBatchNormV3_nhwc;model_787_Conv2D_nhwc;model_516_Conv2D_nhwc')))(conv6_1)
add6_1 = tf.math.add(conv6_2, 3)
relu6_1 = ReLU(max_value=6.0)(add6_1)
div6_1 = tf.math.divide(relu6_1, 6)
mul6_1 = tf.math.multiply(conv5_2, div6_1)

# Block_07
conv7_1 = Conv2D(filters=120, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/model_524_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_525_FusedBatchNormV3_nhwc;model_548_depthwise_nhwc;model_524_Conv2D_nhwc')))(mul6_1)
pad7_1 = tf.pad(conv7_1, paddings=np.load('weights/model_776_pad_Pad_paddings')[[0,2,3,1], :])
depthconv7_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="valid", depth_multiplier=1, dilation_rate=[1, 1], activation='relu',
                 depthwise_initializer=Constant(np.load('weights/model_528_FusedBatchNormV3_nhwc;model_527_depthwise_nhwc;model_548_depthwise_nhwc')),
                 bias_initializer=Constant(np.load('weights/model_528_FusedBatchNormV3_nhwc')))(pad7_1)
conv7_2 = Conv2D(filters=40, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_530_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_531_FusedBatchNormV3_nhwc;model_787_Conv2D_nhwc;model_530_Conv2D_nhwc')))(depthconv7_1)

# Block_08
mean8_1 = mean(conv7_2, axis=[1, 2], keepdims=False)
expand8_1 = expand_dims(mean8_1, axis=1)
expand8_2 = expand_dims(expand8_1, axis=1)
conv8_1 = Conv2D(filters=10, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/model_533_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_534_FusedBatchNormV3_nhwc;model_landmark_Conv2D_nhwc;model_533_Conv2D_nhwc')))(expand8_2)
conv8_2 = Conv2D(filters=40, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_536_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_537_FusedBatchNormV3_nhwc;model_787_Conv2D_nhwc;model_536_Conv2D_nhwc')))(conv8_1)
add8_1 = tf.math.add(conv8_2, 3)
relu8_1 = ReLU(max_value=6.0)(add8_1)
div8_1 = tf.math.divide(relu8_1, 6)
mul8_1 = tf.math.multiply(conv7_2, div8_1)
add8_2 = Add()([mul6_1, mul8_1])

# Block_09
conv9_1 = Conv2D(filters=120, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/model_545_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_546_FusedBatchNormV3_nhwc;model_548_depthwise_nhwc;model_545_Conv2D_nhwc')))(add8_2)
pad9_1 = tf.pad(conv9_1, paddings=np.load('weights/model_776_pad_Pad_paddings')[[0,2,3,1], :])
depthconv9_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="valid", depth_multiplier=1, dilation_rate=[1, 1], activation='relu',
                 depthwise_initializer=Constant(np.load('weights/model_549_FusedBatchNormV3_nhwc;model_548_depthwise_nhwc')),
                 bias_initializer=Constant(np.load('weights/model_549_FusedBatchNormV3_nhwc')))(pad9_1)
conv9_2 = Conv2D(filters=40, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_551_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_552_FusedBatchNormV3_nhwc;model_787_Conv2D_nhwc;model_551_Conv2D_nhwc')))(depthconv9_1)

# Block_10
mean10_1 = mean(conv9_2, axis=[1, 2], keepdims=False)
expand10_1 = expand_dims(mean10_1, axis=1)
expand10_2 = expand_dims(expand10_1, axis=1)
conv10_1 = Conv2D(filters=10, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/model_533_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_534_FusedBatchNormV3_nhwc;model_landmark_Conv2D_nhwc;model_533_Conv2D_nhwc')))(expand10_2)
conv10_2 = Conv2D(filters=40, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_536_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_537_FusedBatchNormV3_nhwc;model_787_Conv2D_nhwc;model_536_Conv2D_nhwc')))(conv10_1)
add10_1 = tf.math.add(conv10_2, 3)
relu10_1 = ReLU(max_value=6.0)(add10_1)
div10_1 = tf.math.divide(relu10_1, 6)
mul10_1 = tf.math.multiply(conv9_2, div10_1)
add10_2 = Add()([add8_2, mul10_1])

# Block_11
conv11_1 = Conv2D(filters=240, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_566_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_567_FusedBatchNormV3_nhwc;model_574_depthwise_nhwc;model_566_Conv2D_nhwc')))(add10_2)
add11_1 = tf.math.add(conv11_1, 3)
relu11_1 = ReLU(max_value=6.0)(add11_1)
mul11_1 = tf.math.multiply(conv11_1, relu11_1)
div11_1 = tf.math.divide(mul11_1, 6)
pad11_1 = tf.pad(div11_1, paddings=np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :])
depthconv11_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding="valid", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/model_575_FusedBatchNormV3_nhwc;model_574_depthwise_nhwc')),
                 bias_initializer=Constant(np.load('weights/model_575_FusedBatchNormV3_nhwc')))(pad11_1)
add11_2 = tf.math.add(depthconv11_1, 3)
relu11_2 = ReLU(max_value=6.0)(add11_2)
mul11_2 = tf.math.multiply(depthconv11_1, relu11_2)
div11_2 = tf.math.divide(mul11_2, 6)
conv11_2 = Conv2D(filters=80, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_582_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_583_FusedBatchNormV3_nhwc;model_638_Conv2D_nhwc;model_582_Conv2D_nhwc')))(div11_2)

# Block_12
conv12_1 = Conv2D(filters=200, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_584_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_585_FusedBatchNormV3_nhwc;model_592_depthwise_nhwc;model_584_Conv2D_nhwc')))(conv11_2)
add12_1 = tf.math.add(conv12_1, 3)
relu12_1 = ReLU(max_value=6.0)(add12_1)
mul12_1 = tf.math.multiply(conv12_1, relu12_1)
div12_1 = tf.math.divide(mul12_1, 6)
pad12_1 = tf.pad(div12_1, paddings=np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :])
depthconv12_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="valid", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/model_593_FusedBatchNormV3_nhwc;model_592_depthwise_nhwc')),
                 bias_initializer=Constant(np.load('weights/model_593_FusedBatchNormV3_nhwc')))(pad12_1)
add12_2 = tf.math.add(depthconv12_1, 3)
relu12_2 = ReLU(max_value=6.0)(add12_2)
mul12_2 = tf.math.multiply(depthconv12_1, relu12_2)
div12_2 = tf.math.divide(mul12_2, 6)
conv12_2 = Conv2D(filters=80, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_600_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_601_FusedBatchNormV3_nhwc;model_638_Conv2D_nhwc;model_600_Conv2D_nhwc')))(div12_2)
add12_3 = Add()([conv11_2, conv12_2])

# Block_13
conv13_1 = Conv2D(filters=184, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_603_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_604_FusedBatchNormV3_nhwc;model_630_depthwise_nhwc;model_603_Conv2D_nhwc')))(add12_3)
add13_1 = tf.math.add(conv13_1, 3)
relu13_1 = ReLU(max_value=6.0)(add13_1)
mul13_1 = tf.math.multiply(conv13_1, relu13_1)
div13_1 = tf.math.divide(mul13_1, 6)
pad13_1 = tf.pad(div13_1, paddings=np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :])
depthconv13_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="valid", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/model_612_FusedBatchNormV3_nhwc;model_611_depthwise_nhwc;model_630_depthwise_nhwc')),
                 bias_initializer=Constant(np.load('weights/model_612_FusedBatchNormV3_nhwc')))(pad13_1)
add13_2 = tf.math.add(depthconv13_1, 3)
relu13_2 = ReLU(max_value=6.0)(add13_2)
mul13_2 = tf.math.multiply(depthconv13_1, relu13_2)
div13_2 = tf.math.divide(mul13_2, 6)
conv13_2 = Conv2D(filters=80, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_619_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_620_FusedBatchNormV3_nhwc;model_638_Conv2D_nhwc;model_619_Conv2D_nhwc')))(div13_2)
add13_3 = Add()([add12_3, conv13_2])

# Block_14
conv14_1 = Conv2D(filters=184, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_622_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_623_FusedBatchNormV3_nhwc;model_630_depthwise_nhwc;model_622_Conv2D_nhwc')))(add13_3)
add14_1 = tf.math.add(conv14_1, 3)
relu14_1 = ReLU(max_value=6.0)(add14_1)
mul14_1 = tf.math.multiply(conv14_1, relu14_1)
div14_1 = tf.math.divide(mul14_1, 6)
pad14_1 = tf.pad(div14_1, paddings=np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :])
depthconv14_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="valid", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/model_631_FusedBatchNormV3_nhwc;model_630_depthwise_nhwc')),
                 bias_initializer=Constant(np.load('weights/model_631_FusedBatchNormV3_nhwc')))(pad14_1)
add14_2 = tf.math.add(depthconv14_1, 3)
relu14_2 = ReLU(max_value=6.0)(add14_2)
mul14_2 = tf.math.multiply(depthconv14_1, relu14_2)
div14_2 = tf.math.divide(mul14_2, 6)
conv14_2 = Conv2D(filters=80, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_638_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_639_FusedBatchNormV3_nhwc;model_638_Conv2D_nhwc')))(div14_2)
add14_3 = Add()([add13_3, conv14_2])

# Block_15
conv15_1 = Conv2D(filters=480, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_641_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_642_FusedBatchNormV3_nhwc;model_649_depthwise_nhwc;model_641_Conv2D_nhwc')))(add14_3)
add15_1 = tf.math.add(conv15_1, 3)
relu15_1 = ReLU(max_value=6.0)(add15_1)
mul15_1 = tf.math.multiply(conv15_1, relu15_1)
div15_1 = tf.math.divide(mul15_1, 6)
pad15_1 = tf.pad(div15_1, paddings=np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :])
depthconv15_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="valid", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/model_650_FusedBatchNormV3_nhwc;model_649_depthwise_nhwc')),
                 bias_initializer=Constant(np.load('weights/model_650_FusedBatchNormV3_nhwc')))(pad15_1)
add15_2 = tf.math.add(depthconv15_1, 3)
relu15_2 = ReLU(max_value=6.0)(add15_2)
mul15_2 = tf.math.multiply(depthconv15_1, relu15_2)
div15_2 = tf.math.divide(mul15_2, 6)
conv15_2 = Conv2D(filters=112, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_657_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_658_FusedBatchNormV3_nhwc;model_696_Conv2D_nhwc;model_657_Conv2D_nhwc')))(div15_2)

# Block_16
mean16_1 = mean(conv15_2, axis=[1, 2], keepdims=False)
expand16_1 = expand_dims(mean16_1, axis=1)
expand16_2 = expand_dims(expand16_1, axis=1)
conv16_1 = Conv2D(filters=28, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/model_660_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_661_FusedBatchNormV3_nhwc;model_693_Conv2D_nhwc;model_660_Conv2D_nhwc')))(expand16_2)
conv16_2 = Conv2D(filters=112, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_663_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_664_FusedBatchNormV3_nhwc;model_696_Conv2D_nhwc;model_663_Conv2D_nhwc')))(conv16_1)
add16_1 = tf.math.add(conv16_2, 3)
relu16_1 = ReLU(max_value=6.0)(add16_1)
div16_1 = tf.math.divide(relu16_1, 6)
mul16_1 = tf.math.multiply(conv15_2, div16_1)

conv16_3 = Conv2D(filters=112, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_671_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_672_FusedBatchNormV3_nhwc;model_696_Conv2D_nhwc;model_671_Conv2D_nhwc')))(add14_3)

add16_2 = Add()([mul16_1, conv16_3])

# Block_17
conv17_1 = Conv2D(filters=672, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_674_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_675_FusedBatchNormV3_nhwc;model_746_depthwise_nhwc;model_674_Conv2D_nhwc')))(add16_2)
add17_1 = tf.math.add(conv17_1, 3)
relu17_1 = ReLU(max_value=6.0)(add17_1)
mul17_1 = tf.math.multiply(conv17_1, relu17_1)
div17_1 = tf.math.divide(mul17_1, 6)
pad17_1 = tf.pad(div17_1, paddings=np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :])
depthconv17_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="valid", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/model_683_FusedBatchNormV3_nhwc;model_682_depthwise_nhwc;model_746_depthwise_nhwc')),
                 bias_initializer=Constant(np.load('weights/model_683_FusedBatchNormV3_nhwc')))(pad17_1)
add17_2 = tf.math.add(depthconv17_1, 3)
relu17_2 = ReLU(max_value=6.0)(add17_2)
mul17_2 = tf.math.multiply(depthconv17_1, relu17_2)
div17_2 = tf.math.divide(mul17_2, 6)
conv17_2 = Conv2D(filters=112, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_690_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_691_FusedBatchNormV3_nhwc;model_696_Conv2D_nhwc;model_690_Conv2D_nhwc')))(div17_2)

# Block_18
mean18_1 = mean(conv17_2, axis=[1, 2], keepdims=False)
expand18_1 = expand_dims(mean18_1, axis=1)
expand18_2 = expand_dims(expand18_1, axis=1)
conv18_1 = Conv2D(filters=28, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/model_693_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_694_FusedBatchNormV3_nhwc;model_693_Conv2D_nhwc')))(expand18_2)
conv18_2 = Conv2D(filters=112, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_696_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_697_FusedBatchNormV3_nhwc;model_696_Conv2D_nhwc')))(conv18_1)
add18_1 = tf.math.add(conv18_2, 3)
relu18_1 = ReLU(max_value=6.0)(add18_1)
div18_1 = tf.math.divide(relu18_1, 6)
mul18_1 = tf.math.multiply(conv17_2, div18_1)
add18_2 = Add()([add16_2, mul18_1])

# Block_19
conv19_1 = Conv2D(filters=672, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_705_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_706_FusedBatchNormV3_nhwc;model_746_depthwise_nhwc;model_705_Conv2D_nhwc')))(add18_2)
add19_1 = tf.math.add(conv19_1, 3)
relu19_1 = ReLU(max_value=6.0)(add19_1)
mul19_1 = tf.math.multiply(conv19_1, relu19_1)
div19_1 = tf.math.divide(mul19_1, 6)
pad19_1 = tf.pad(div19_1, paddings=np.load('weights/model_776_pad_Pad_paddings')[[0,2,3,1], :])
depthconv19_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="valid", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/model_714_FusedBatchNormV3_nhwc;model_713_depthwise_nhwc;model_746_depthwise_nhwc')),
                 bias_initializer=Constant(np.load('weights/model_714_FusedBatchNormV3_nhwc')))(pad19_1)
add19_2 = tf.math.add(depthconv19_1, 3)
relu19_2 = ReLU(max_value=6.0)(add19_2)
mul19_2 = tf.math.multiply(depthconv19_1, relu19_2)
div19_2 = tf.math.divide(mul19_2, 6)
conv19_2 = Conv2D(filters=160, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_721_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_722_FusedBatchNormV3_nhwc;model_790_Conv2D_nhwc;model_721_Conv2D_nhwc')))(div19_2)

# Block_20
mean20_1 = mean(conv19_2, axis=[1, 2], keepdims=False)
expand20_1 = expand_dims(mean20_1, axis=1)
expand20_2 = expand_dims(expand20_1, axis=1)
conv20_1 = Conv2D(filters=40, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/model_724_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_725_FusedBatchNormV3_nhwc;model_787_Conv2D_nhwc;model_724_Conv2D_nhwc')))(expand20_2)
conv20_2 = Conv2D(filters=160, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_727_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_728_FusedBatchNormV3_nhwc;model_790_Conv2D_nhwc;model_727_Conv2D_nhwc')))(conv20_1)
add20_1 = tf.math.add(conv20_2, 3)
relu20_1 = ReLU(max_value=6.0)(add20_1)
div20_1 = tf.math.divide(relu20_1, 6)
mul20_1 = tf.math.multiply(conv19_2, div20_1)

conv20_3 = Conv2D(filters=160, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_735_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_736_FusedBatchNormV3_nhwc;model_790_Conv2D_nhwc;model_735_Conv2D_nhwc')))(add18_2)

add20_2 = Add()([mul20_1, conv20_3])

# Block_21
conv21_1 = Conv2D(filters=672, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_738_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_739_FusedBatchNormV3_nhwc;model_746_depthwise_nhwc;model_738_Conv2D_nhwc')))(add20_2)
add21_1 = tf.math.add(conv21_1, 3)
relu21_1 = ReLU(max_value=6.0)(add21_1)
mul21_1 = tf.math.multiply(conv21_1, relu21_1)
div21_1 = tf.math.divide(mul21_1, 6)
pad21_1 = tf.pad(div21_1, paddings=np.load('weights/model_776_pad_Pad_paddings')[[0,2,3,1], :])
depthconv21_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[2, 2], padding="valid", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/model_747_FusedBatchNormV3_nhwc;model_746_depthwise_nhwc')),
                 bias_initializer=Constant(np.load('weights/model_747_FusedBatchNormV3_nhwc')))(pad21_1)
add21_2 = tf.math.add(depthconv21_1, 3)
relu21_2 = ReLU(max_value=6.0)(add21_2)
mul21_2 = tf.math.multiply(depthconv21_1, relu21_2)
div21_2 = tf.math.divide(mul21_2, 6)
conv21_2 = Conv2D(filters=160, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_754_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_755_FusedBatchNormV3_nhwc;model_790_Conv2D_nhwc;model_754_Conv2D_nhwc')))(div21_2)

# Block_22
mean22_1 = mean(conv21_2, axis=[1, 2], keepdims=False)
expand22_1 = expand_dims(mean22_1, axis=1)
expand22_2 = expand_dims(expand22_1, axis=1)
conv22_1 = Conv2D(filters=40, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/model_757_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_758_FusedBatchNormV3_nhwc;model_787_Conv2D_nhwc;model_757_Conv2D_nhwc')))(expand22_2)
conv22_2 = Conv2D(filters=160, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_760_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_761_FusedBatchNormV3_nhwc;model_790_Conv2D_nhwc;model_760_Conv2D_nhwc')))(conv22_1)
add22_1 = tf.math.add(conv22_2, 3)
relu22_1 = ReLU(max_value=6.0)(add22_1)
div22_1 = tf.math.divide(relu22_1, 6)
mul22_1 = tf.math.multiply(conv21_2, div22_1)

# Block_23
conv23_1 = Conv2D(filters=960, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_768_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_769_FusedBatchNormV3_nhwc;model_799_Conv2D_nhwc;model_768_Conv2D_nhwc')))(mul22_1)
add23_1 = tf.math.add(conv23_1, 3)
relu23_1 = ReLU(max_value=6.0)(add23_1)
mul23_1 = tf.math.multiply(conv23_1, relu23_1)
div23_1 = tf.math.divide(mul23_1, 6)
pad23_1 = tf.pad(div23_1, paddings=np.load('weights/model_776_pad_Pad_paddings')[[0,2,3,1], :])
depthconv23_1 = DepthwiseConv2D(kernel_size=[5, 5], strides=[1, 1], padding="valid", depth_multiplier=1, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights/model_777_FusedBatchNormV3_nhwc;model_776_depthwise_nhwc;model_799_Conv2D_nhwc')),
                 bias_initializer=Constant(np.load('weights/model_777_FusedBatchNormV3_nhwc')))(pad23_1)
add23_2 = tf.math.add(depthconv23_1, 3)
relu23_2 = ReLU(max_value=6.0)(add23_2)
mul23_2 = tf.math.multiply(depthconv23_1, relu23_2)
div23_2 = tf.math.divide(mul23_2, 6)
conv23_2 = Conv2D(filters=160, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_784_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_785_FusedBatchNormV3_nhwc;model_790_Conv2D_nhwc;model_784_Conv2D_nhwc')))(div23_2)

# Block_24
mean24_1 = mean(conv23_2, axis=[1, 2], keepdims=False)
expand24_1 = expand_dims(mean24_1, axis=1)
expand24_2 = expand_dims(expand24_1, axis=1)
conv24_1 = Conv2D(filters=40, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/model_787_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_788_FusedBatchNormV3_nhwc;model_787_Conv2D_nhwc')))(expand24_2)
conv24_2 = Conv2D(filters=160, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_790_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_791_FusedBatchNormV3_nhwc;model_790_Conv2D_nhwc')))(conv24_1)
add24_1 = tf.math.add(conv24_2, 3)
relu24_1 = ReLU(max_value=6.0)(add24_1)
div24_1 = tf.math.divide(relu24_1, 6)
mul24_1 = tf.math.multiply(conv23_2, div24_1)
add24_2 = Add()([mul22_1, mul24_1])

# Block_25
conv25_1 = Conv2D(filters=960, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_799_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_800_FusedBatchNormV3_nhwc;model_799_Conv2D_nhwc')))(add24_2)
add25_1 = tf.math.add(conv25_1, 3)
relu25_1 = ReLU(max_value=6.0)(add25_1)
mul25_1 = tf.math.multiply(conv25_1, relu25_1)
div25_1 = tf.math.divide(mul25_1, 6)

conv25_2 = Conv2D(filters=320, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_807_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_808_FusedBatchNormV3_nhwc;model_807_Conv2D_nhwc')))(div25_1)
add25_2 = tf.math.add(conv25_2, 3)
relu25_2 = ReLU(max_value=6.0)(add25_2)
mul25_2 = tf.math.multiply(conv25_2, relu25_2)
div25_2 = tf.math.divide(mul25_2, 6)

conv25_3 = Conv2D(filters=24, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_815_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_816_FusedBatchNormV3_nhwc;model_963_Conv2D_nhwc;model_815_Conv2D_nhwc')))(div25_2)
add25_3 = tf.math.add(conv25_3, 3)
relu25_3 = ReLU(max_value=6.0)(add25_3)
mul25_3 = tf.math.multiply(conv25_3, relu25_3)
div25_3 = tf.math.divide(mul25_3, 6)

# Block_26
shape26_1 = tf.shape(div25_3, out_type=tf.dtypes.int32)
stride26_1 = tf.strided_slice(shape26_1,
                              begin=[1],
                              end=[3],
                              strides=[1],
                              begin_mask=0,
                              end_mask=0,
                              ellipsis_mask=0,
                              new_axis_mask=0,
                              shrink_axis_mask=0)
mul26_1 = tf.math.multiply(stride26_1, np.load('weights/model_937_Const'))
resize26_1 = tf.image.resize(div25_3, mul26_1, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# Block_27
pad27_1 = tf.pad(resize26_1, paddings=np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :])
conv27_1 = Conv2D(filters=24, kernel_size=[3, 3], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_850_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_851_FusedBatchNormV3_nhwc;model_963_Conv2D_nhwc;model_850_Conv2D_nhwc')))(pad27_1)
add27_1 = tf.math.add(conv27_1, 3)
relu27_1 = ReLU(max_value=6.0)(add27_1)
mul27_1 = tf.math.multiply(conv27_1, relu27_1)
div27_1 = tf.math.divide(mul27_1, 6)

conv27_2 = Conv2D(filters=24, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_858_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_859_FusedBatchNormV3_nhwc;model_963_Conv2D_nhwc;model_858_Conv2D_nhwc')))(add20_2)
add27_2 = tf.math.add(conv27_2, 3)
relu27_2 = ReLU(max_value=6.0)(add27_2)
mul27_2 = tf.math.multiply(conv27_2, relu27_2)
div27_2 = tf.math.divide(mul27_2, 6)

add27_3 = Add()([div27_1, div27_2])

# Block_28
shape28_1 = tf.shape(add27_3, out_type=tf.dtypes.int32)
stride28_1 = tf.strided_slice(shape28_1,
                              begin=[1],
                              end=[3],
                              strides=[1],
                              begin_mask=0,
                              end_mask=0,
                              ellipsis_mask=0,
                              new_axis_mask=0,
                              shrink_axis_mask=0)
mul28_1 = tf.math.multiply(stride28_1, np.load('weights/model_937_Const'))
resize28_1 = tf.image.resize(add27_3, mul28_1, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# Block_29
pad29_1 = tf.pad(resize28_1, paddings=np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :])
conv29_1 = Conv2D(filters=24, kernel_size=[3, 3], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_894_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_895_FusedBatchNormV3_nhwc;model_963_Conv2D_nhwc;model_894_Conv2D_nhwc')))(pad29_1)
add29_1 = tf.math.add(conv29_1, 3)
relu29_1 = ReLU(max_value=6.0)(add29_1)
mul29_1 = tf.math.multiply(conv29_1, relu29_1)
div29_1 = tf.math.divide(mul29_1, 6)

conv29_2 = Conv2D(filters=24, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_902_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_903_FusedBatchNormV3_nhwc;model_963_Conv2D_nhwc;model_902_Conv2D_nhwc')))(add10_2)
add29_2 = tf.math.add(conv29_2, 3)
relu29_2 = ReLU(max_value=6.0)(add29_2)
mul29_2 = tf.math.multiply(conv29_2, relu29_2)
div29_2 = tf.math.divide(mul29_2, 6)

add29_3 = Add()([div29_1, div29_2])

# Block_30
shape30_1 = tf.shape(add29_3, out_type=tf.dtypes.int32)
stride30_1 = tf.strided_slice(shape30_1,
                              begin=[1],
                              end=[3],
                              strides=[1],
                              begin_mask=0,
                              end_mask=0,
                              ellipsis_mask=0,
                              new_axis_mask=0,
                              shrink_axis_mask=0)
mul30_1 = tf.math.multiply(stride30_1, np.load('weights/model_937_Const'))
resize30_1 = tf.image.resize(add29_3, mul30_1, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# Block_31
pad31_1 = tf.pad(resize30_1, paddings=np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :])
conv31_1 = Conv2D(filters=24, kernel_size=[3, 3], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_938_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_939_FusedBatchNormV3_nhwc;model_963_Conv2D_nhwc;model_938_Conv2D_nhwc')))(pad31_1)
add31_1 = tf.math.add(conv31_1, 3)
relu31_1 = ReLU(max_value=6.0)(add31_1)
mul31_1 = tf.math.multiply(conv31_1, relu31_1)
div31_1 = tf.math.divide(mul31_1, 6)

conv31_2 = Conv2D(filters=24, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_946_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_947_FusedBatchNormV3_nhwc;model_963_Conv2D_nhwc;model_946_Conv2D_nhwc')))(add4_1)
add31_2 = tf.math.add(conv31_2, 3)
relu31_2 = ReLU(max_value=6.0)(add31_2)
mul31_2 = tf.math.multiply(conv31_2, relu31_2)
div31_2 = tf.math.divide(mul31_2, 6)

add31_3 = Add()([div31_1, div31_2])

# Block_32
pad32_1 = tf.pad(add31_3, paddings=np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :])
conv32_1 = Conv2D(filters=24, kernel_size=[3, 3], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_963_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_964_FusedBatchNormV3_nhwc;model_963_Conv2D_nhwc')))(pad32_1)
add32_1 = tf.math.add(conv32_1, 3)
relu32_1 = ReLU(max_value=6.0)(add32_1)
mul32_1 = tf.math.multiply(conv32_1, relu32_1)
div32_1 = tf.math.divide(mul32_1, 6)

# Block_33
stride33_1 = tf.strided_slice(div32_1,
                              begin=[0,0,0,12],
                              end=[1,120,160,24],
                              strides=[1,1,1,1],
                              begin_mask=0,
                              end_mask=0,
                              ellipsis_mask=0,
                              new_axis_mask=0,
                              shrink_axis_mask=0)
pad33_1 = tf.pad(stride33_1, paddings=np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :])
conv33_1 = Conv2D(filters=12, kernel_size=[3, 3], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_981_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_982_FusedBatchNormV3_nhwc;model_989_Conv2D_nhwc;model_981_Conv2D_nhwc')))(pad33_1)
add33_1 = tf.math.add(conv33_1, 3)
relu33_1 = ReLU(max_value=6.0)(add33_1)
mul33_1 = tf.math.multiply(conv33_1, relu33_1)
div33_1 = tf.math.divide(mul33_1, 6)

pad33_2 = tf.pad(div33_1, paddings=np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :])
conv33_2 = Conv2D(filters=12, kernel_size=[3, 3], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_989_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_990_FusedBatchNormV3_nhwc;model_989_Conv2D_nhwc')))(pad33_2)
add33_2 = tf.math.add(conv33_2, 3)
relu33_2 = ReLU(max_value=6.0)(add33_2)
mul33_2 = tf.math.multiply(conv33_2, relu33_2)
div33_2 = tf.math.divide(mul33_2, 6)

stride33_2 = tf.strided_slice(div32_1,
                              begin=[0,0,0,0],
                              end=[1,120,160,12],
                              strides=[1,1,1,1],
                              begin_mask=0,
                              end_mask=0,
                              ellipsis_mask=0,
                              new_axis_mask=0,
                              shrink_axis_mask=0)
pad33_3 = tf.pad(stride33_2, paddings=np.load('weights/model_989_pad_Pad_paddings')[[0,2,3,1], :])
conv33_3 = Conv2D(filters=12, kernel_size=[3, 3], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_973_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_974_FusedBatchNormV3_nhwc;model_989_Conv2D_nhwc;model_973_Conv2D_nhwc')))(pad33_3)
add33_3 = tf.math.add(conv33_3, 3)
relu33_3 = ReLU(max_value=6.0)(add33_3)
mul33_3 = tf.math.multiply(conv33_3, relu33_3)
div33_3 = tf.math.divide(mul33_3, 6)

concat33_1 = Concatenate(axis=3)([div33_3, div33_2])

conv33_4 = Conv2D(filters=24, kernel_size=[3, 3], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_955_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_956_FusedBatchNormV3_nhwc;model_963_Conv2D_nhwc;model_955_Conv2D_nhwc')))(pad32_1)
add33_4 = tf.math.add(conv33_4, 3)
relu33_4 = ReLU(max_value=6.0)(add33_4)
mul33_4 = tf.math.multiply(conv33_4, relu33_4)
div33_4 = tf.math.divide(mul33_4, 6)

concat33_2 = Concatenate(axis=3)([div33_4, concat33_1])

# Block_34
conv34_1 = Conv2D(filters=4, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_1000_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_1000_BiasAdd_nhwc;model_1000_Conv2D_nhwc;model_1000_BiasAdd_ReadVariableOp_resource')))(concat33_2)
exp34_1 = exp(conv34_1)

conv34_2 = Conv2D(filters=1, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_999_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_999_BiasAdd_nhwc;model_999_Conv2D_nhwc;model_999_BiasAdd_ReadVariableOp_resource')))(concat33_2)
sigm34_1 = tf.math.sigmoid(conv34_2)

conv34_3 = Conv2D(filters=10, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/model_landmark_Conv2D_nhwc').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/model_landmark_BiasAdd_nhwc;model_landmark_Conv2D_nhwc;model_landmark_BiasAdd_ReadVariableOp_resource')))(concat33_2)



model = Model(inputs=inputs, outputs=[exp34_1, sigm34_1, conv34_3])

model.summary()

tf.saved_model.save(model, 'saved_model_keras_480x640')
model.save('dbface_keras_480x640.h5')

full_model = tf.function(lambda inputs: model(inputs))
full_model = full_model.get_concrete_function(inputs = (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)))
frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
frozen_func.graph.as_graph_def()
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=".",
                    name="dbface_keras_480x640_float32_nhwc.pb",
                    as_text=False)


# No Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('dbface_keras_480x640_float32_nhwc.tflite', 'wb') as w:
    w.write(tflite_model)
print("tflite convert complete! - dbface_keras_480x640_float32_nhwc.tflite")


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
with open('dbface_keras_480x640_weight_quant_nhwc.tflite', 'wb') as w:
    w.write(tflite_model)
print("Weight Quantization complete! - dbface_keras_480x640_weight_quant_nhwc.tflite")


def representative_dataset_gen():
  for data in raw_test_data.take(100):
    image = data['image'].numpy()
    image = tf.image.resize(image, (480, 640))
    image = image[np.newaxis,:,:,:]
    image = image - 127.5
    image = image * 0.007843
    yield [image]

raw_test_data, info = tfds.load(name="the300w_lp", with_info=True, split="train", data_dir="~/TFDS", download=False)


# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('dbface_keras_480x640_integer_quant_nhwc.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - dbface_keras_480x640_integer_quant_nhwc.tflite")


# # Full Integer Quantization - Input/Output=int8
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('dbface_keras_480x640_full_integer_quant_nhwc.tflite', 'wb') as w:
#     w.write(tflite_quant_model)
# print("Full Integer Quantization complete! - dbface_keras_480x640_full_integer_quant_nhwc.tflite")


# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('dbface_keras_480x640_float16_quant_nhwc.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - dbface_keras_480x640_float16_quant_nhwc.tflite")


# # EdgeTPU
# import subprocess
# result = subprocess.check_output(["edgetpu_compiler", "-s", "dbface_keras_480x640_full_integer_quant_nhwc.tflite"])
# print(result)
