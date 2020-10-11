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

###  saved_model_cli show --dir saved_model_kitti_192x640/ --tag_set serve --signature_def serving_default

'''
The given SavedModel SignatureDef contains the following input(s):
  inputs['input'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 192, 640, 3)
      name: serving_default_input:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['concatenate_10'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 192, 640, 4)
      name: StatefulPartitionedCall:0
  outputs['concatenate_11'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 192, 640, 4)
      name: StatefulPartitionedCall:1
  outputs['concatenate_6'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 192, 640, 4)
      name: StatefulPartitionedCall:2
  outputs['concatenate_8'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 192, 640, 4)
      name: StatefulPartitionedCall:3
Method name is: tensorflow/serving/predict
'''


import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, ReLU, MaxPool2D, Reshape, Concatenate, ZeroPadding2D, Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.backend import resize_images
from tensorflow.keras.activations import tanh
from tensorflow.math import sigmoid
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import sys
import tensorflow_datasets as tfds

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


height = 192
width  = 640
ds = 'kitti'

inputs = Input(shape=(height, width, 3), batch_size=1, name='input')

# Block_01
mul1_1 = tf.math.multiply(inputs, np.load('weights/{}_{}x{}/FP32/data_mul_13016_copy_const.npy'.format(ds, height, width)).transpose(0,2,3,1))
add1_1 = Add()([mul1_1, np.load('weights/{}_{}x{}/FP32/data_add_13018_copy_const.npy'.format(ds, height, width)).flatten()])
conv1_1 = Conv2D(filters=64, kernel_size=[7, 7], strides=[2, 2], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/490_mean_Fused_Mul_1412714129_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1302113026_copy_const.npy'.format(ds, height, width)).flatten()))(add1_1)

# Block_02
maxpool2_1 = tf.nn.max_pool(conv1_1, ksize=[3, 3], strides=[2, 2], padding='SAME')
conv2_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/494_mean_Fused_Mul_1413114133_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1302913034_copy_const.npy'.format(ds, height, width)).flatten()))(maxpool2_1)
conv2_2 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/497_mean_Fused_Mul_1413514137_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1303713042_copy_const.npy'.format(ds, height, width)).flatten()))(conv2_1)
add2_1 = Add()([conv2_2, maxpool2_1])
relu2_1 = ReLU()(add2_1)

# Block_03
conv3_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/501_mean_Fused_Mul_1413914141_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1304513050_copy_const.npy'.format(ds, height, width)).flatten()))(relu2_1)
conv3_2 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/504_mean_Fused_Mul_1414314145_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1305313058_copy_const.npy'.format(ds, height, width)).flatten()))(conv3_1)
add3_1 = Add()([conv3_2, relu2_1])
relu3_1 = ReLU()(add3_1)

# Block_04
conv4_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/508_mean_Fused_Mul_1414714149_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1306113066_copy_const.npy'.format(ds, height, width)).flatten()))(relu3_1)
conv4_2 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/511_mean_Fused_Mul_1415114153_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1306913074_copy_const.npy'.format(ds, height, width)).flatten()))(conv4_1)
add4_1 = Add()([conv4_2, relu3_1])
relu4_1 = ReLU()(add4_1)

# Block_05
conv5_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[2, 2], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/515_mean_Fused_Mul_1415914161_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1308513090_copy_const.npy'.format(ds, height, width)).flatten()))(relu4_1)
conv5_2 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/518_mean_Fused_Mul_1416314165_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1309313098_copy_const.npy'.format(ds, height, width)).flatten()))(conv5_1)

conv5_3 = Conv2D(filters=128, kernel_size=[1, 1], strides=[2, 2], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/520_mean_Fused_Mul_1415514157_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1307713082_copy_const.npy'.format(ds, height, width)).flatten()))(relu4_1)

add5_1 = Add()([conv5_2, conv5_3])
relu5_1 = ReLU()(add5_1)

# Block_06
conv6_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/524_mean_Fused_Mul_1416714169_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1310113106_copy_const.npy'.format(ds, height, width)).flatten()))(relu5_1)
conv6_2 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/527_mean_Fused_Mul_1417114173_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1310913114_copy_const.npy'.format(ds, height, width)).flatten()))(conv6_1)
add6_1 = Add()([conv6_2, relu5_1])
relu6_1 = ReLU()(add6_1)

# Block_07
conv7_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/531_mean_Fused_Mul_1417514177_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1311713122_copy_const.npy'.format(ds, height, width)).flatten()))(relu6_1)
conv7_2 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/534_mean_Fused_Mul_1417914181_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1312513130_copy_const.npy'.format(ds, height, width)).flatten()))(conv7_1)
add7_1 = Add()([conv7_2, relu6_1])
relu7_1 = ReLU()(add7_1)

# Block_08
conv8_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/538_mean_Fused_Mul_1418314185_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1313313138_copy_const.npy'.format(ds, height, width)).flatten()))(relu7_1)
conv8_2 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/541_mean_Fused_Mul_1418714189_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1314113146_copy_const.npy'.format(ds, height, width)).flatten()))(conv8_1)
add8_1 = Add()([conv8_2, relu7_1])
relu8_1 = ReLU()(add8_1)

# Block_09
conv9_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[2, 2], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/545_mean_Fused_Mul_1419514197_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1315713162_copy_const.npy'.format(ds, height, width)).flatten()))(relu8_1)
conv9_2 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/548_mean_Fused_Mul_1419914201_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1316513170_copy_const.npy'.format(ds, height, width)).flatten()))(conv9_1)

conv9_3 = Conv2D(filters=256, kernel_size=[1, 1], strides=[2, 2], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/550_mean_Fused_Mul_1419114193_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1314913154_copy_const.npy'.format(ds, height, width)).flatten()))(relu8_1)

add9_1 = Add()([conv9_2, conv9_3])
relu9_1 = ReLU()(add9_1)

# Block_10
conv10_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/554_mean_Fused_Mul_1420314205_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1317313178_copy_const.npy'.format(ds, height, width)).flatten()))(relu9_1)
conv10_2 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/557_mean_Fused_Mul_1420714209_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1318113186_copy_const.npy'.format(ds, height, width)).flatten()))(conv10_1)
add10_1 = Add()([conv10_2, relu9_1])
relu10_1 = ReLU()(add10_1)

# Block_11
conv11_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/561_mean_Fused_Mul_1421114213_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1318913194_copy_const.npy'.format(ds, height, width)).flatten()))(relu10_1)
conv11_2 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/564_mean_Fused_Mul_1421514217_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1319713202_copy_const.npy'.format(ds, height, width)).flatten()))(conv11_1)
add11_1 = Add()([conv11_2, relu10_1])
relu11_1 = ReLU()(add11_1)

# Block_12
conv12_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/568_mean_Fused_Mul_1421914221_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1320513210_copy_const.npy'.format(ds, height, width)).flatten()))(relu11_1)
conv12_2 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/571_mean_Fused_Mul_1422314225_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1321313218_copy_const.npy'.format(ds, height, width)).flatten()))(conv12_1)
add12_1 = Add()([conv12_2, relu11_1])
relu12_1 = ReLU()(add12_1)

# Block_13
conv13_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/575_mean_Fused_Mul_1422714229_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1322113226_copy_const.npy'.format(ds, height, width)).flatten()))(relu12_1)
conv13_2 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/578_mean_Fused_Mul_1423114233_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1322913234_copy_const.npy'.format(ds, height, width)).flatten()))(conv13_1)
add13_1 = Add()([conv13_2, relu12_1])
relu13_1 = ReLU()(add13_1)

# Block_14
conv14_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/582_mean_Fused_Mul_1423514237_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1323713242_copy_const.npy'.format(ds, height, width)).flatten()))(relu13_1)
conv14_2 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/585_mean_Fused_Mul_1423914241_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1324513250_copy_const.npy'.format(ds, height, width)).flatten()))(conv14_1)
add14_1 = Add()([conv14_2, relu13_1])
relu14_1 = ReLU()(add14_1)

# Block_15
conv15_1 = Conv2D(filters=512, kernel_size=[3, 3], strides=[2, 2], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/589_mean_Fused_Mul_1424714249_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1326113266_copy_const.npy'.format(ds, height, width)).flatten()))(relu14_1)
conv15_2 = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/592_mean_Fused_Mul_1425114253_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1326913274_copy_const.npy'.format(ds, height, width)).flatten()))(conv15_1)

conv15_3 = Conv2D(filters=512, kernel_size=[1, 1], strides=[2, 2], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/594_mean_Fused_Mul_1424314245_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1325313258_copy_const.npy'.format(ds, height, width)).flatten()))(relu14_1)

add15_1 = Add()([conv15_2, conv15_3])
relu15_1 = ReLU()(add15_1)

# Block_16
conv16_1 = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/598_mean_Fused_Mul_1425514257_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1327713282_copy_const.npy'.format(ds, height, width)).flatten()))(relu15_1)
conv16_2 = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/601_mean_Fused_Mul_1425914261_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1328513290_copy_const.npy'.format(ds, height, width)).flatten()))(conv16_1)
add16_1 = Add()([conv16_2, relu15_1])
relu16_1 = ReLU()(add16_1)

# Block_17
conv17_1 = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/605_mean_Fused_Mul_1426314265_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1329313298_copy_const.npy'.format(ds, height, width)).flatten()))(relu16_1)
conv17_2 = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/608_mean_Fused_Mul_1426714269_const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/data_add_1330113306_copy_const.npy'.format(ds, height, width)).flatten()))(conv17_1)
add17_1 = Add()([conv17_2, relu16_1])
relu17_1 = ReLU()(add17_1)



# Block_18 ####################################################################################################################
conv18_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.block1.pre_concat_conv.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/612_Dims7877_copy_const.npy'.format(ds, height, width)).flatten()))(relu17_1)
conv18_2 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.block1.pre_concat_conv.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/615_Dims7733_copy_const.npy'.format(ds, height, width)).flatten()))(conv18_1)
resize18_1 = resize_images(conv18_2, 2, 2, 'channels_last', interpolation='nearest')
concat18_1 = Concatenate()([resize18_1, relu14_1])


conv18_3 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.block1.post_concat_conv.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/628_Dims7787_copy_const.npy'.format(ds, height, width)).flatten()))(concat18_1)
conv18_4 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.block1.post_concat_conv.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/631_Dims7859_copy_const.npy'.format(ds, height, width)).flatten()))(conv18_3)
conv18_5 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.block2.pre_concat_conv.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/634_Dims7805_copy_const.npy'.format(ds, height, width)).flatten()))(conv18_4)
conv18_6 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.block2.pre_concat_conv.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/637_Dims7913_copy_const.npy'.format(ds, height, width)).flatten()))(conv18_5)
resize18_2 = resize_images(conv18_6, 2, 2, 'channels_last', interpolation='nearest')
concat18_2 = Concatenate()([resize18_2, relu8_1])

conv18_7 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.block2.post_concat_conv.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/650_Dims7895_copy_const.npy'.format(ds, height, width)).flatten()))(concat18_2)
conv18_8 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.block2.post_concat_conv.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/653_Dims7799_copy_const.npy'.format(ds, height, width)).flatten()))(conv18_7)

# Block_19
conv19_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.block3.pre_concat_conv.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/667_Dims7865_copy_const.npy'.format(ds, height, width)).flatten()))(conv18_8)
conv19_2 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.block3.pre_concat_conv.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/670_Dims7781_copy_const.npy'.format(ds, height, width)).flatten()))(conv19_1)
resize19_1 = resize_images(conv19_2, 2, 2, 'channels_last', interpolation='nearest')
concat19_1 = Concatenate()([resize19_1, relu4_1])

conv19_3 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.block3.post_concat_conv.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/683_Dims7907_copy_const.npy'.format(ds, height, width)).flatten()))(concat19_1)

conv19_4 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.block3.post_concat_conv.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/686_Dims7793_copy_const.npy'.format(ds, height, width)).flatten()))(conv19_3)

conv19_5 = Conv2D(filters=2, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.outconv2.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/689_Dims7829_copy_const.npy'.format(ds, height, width)).flatten()))(conv19_4)
resize19_2 = resize_images(conv19_5, 4, 4, 'channels_last', interpolation='bilinear')


# Block_20
conv20_1 = Conv2D(filters=2, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.outconv1.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/656_Dims7667_copy_const.npy'.format(ds, height, width)).flatten()))(conv18_8)
resize20_1 = resize_images(conv20_1, 8, 8, 'channels_last', interpolation='nearest')

# Block_21
conv21_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.block4.pre_concat_conv.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/700_Dims7751_copy_const.npy'.format(ds, height, width)).flatten()))(conv19_4)
conv21_2 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.block4.pre_concat_conv.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/703_Dims7691_copy_const.npy'.format(ds, height, width)).flatten()))(conv21_1)
resize21_1 = resize_images(conv21_2, 2, 2, 'channels_last', interpolation='nearest')
concat21_1 = Concatenate()([resize21_1, conv1_1])

# Block_22
conv22_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.block4.post_concat_conv.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/716_Dims7901_copy_const.npy'.format(ds, height, width)).flatten()))(concat21_1)

conv22_2 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.block4.post_concat_conv.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/719_Dims7919_copy_const.npy'.format(ds, height, width)).flatten()))(conv22_1)


# Block_50 ####################################################################################################################
conv50_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.block1.pre_concat_conv.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/750_Dims7763_copy_const.npy'.format(ds, height, width)).flatten()))(relu17_1)
conv50_2 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.block1.pre_concat_conv.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/753_Dims7721_copy_const.npy'.format(ds, height, width)).flatten()))(conv50_1)
resize50_1 = resize_images(conv50_2, 2, 2, 'channels_last', interpolation='nearest')
concat50_1 = Concatenate()([resize50_1, relu14_1])


conv50_3 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.block1.post_concat_conv.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/766_Dims7739_copy_const.npy'.format(ds, height, width)).flatten()))(concat50_1)
conv50_4 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.block1.post_concat_conv.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/769_Dims7679_copy_const.npy'.format(ds, height, width)).flatten()))(conv50_3)
conv50_5 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.block2.pre_concat_conv.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/772_Dims7835_copy_const.npy'.format(ds, height, width)).flatten()))(conv50_4)
conv50_6 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.block2.pre_concat_conv.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/775_Dims7715_copy_const.npy'.format(ds, height, width)).flatten()))(conv50_5)
resize50_2 = resize_images(conv50_6, 2, 2, 'channels_last', interpolation='nearest')
concat50_2 = Concatenate()([resize50_2, relu8_1])

conv50_7 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.block2.post_concat_conv.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/788_Dims7817_copy_const.npy'.format(ds, height, width)).flatten()))(concat50_2)
conv50_8 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.block2.post_concat_conv.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/791_Dims7889_copy_const.npy'.format(ds, height, width)).flatten()))(conv50_7)

conv50_9 = Conv2D(filters=2, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.outconv1.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/794_Dims7775_copy_const.npy'.format(ds, height, width)).flatten()))(conv50_8)
sigm50_1 = sigmoid(conv50_9)
resize50_3 = resize_images(sigm50_1, 8, 8, 'channels_last', interpolation='nearest')

concat50_3 = Concatenate()([resize20_1, resize50_3])



conv50_10 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.block3.pre_concat_conv.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/806_Dims7769_copy_const.npy'.format(ds, height, width)).flatten()))(conv50_8)
conv50_11 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.block3.pre_concat_conv.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/809_Dims7811_copy_const.npy'.format(ds, height, width)).flatten()))(conv50_10)
resize50_4 = resize_images(conv50_11, 2, 2, 'channels_last', interpolation='nearest')

concat50_4 = Concatenate()([resize50_4, relu4_1])

conv50_12 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.block3.post_concat_conv.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/822_Dims7685_copy_const.npy'.format(ds, height, width)).flatten()))(concat50_4)
conv50_13 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.block3.post_concat_conv.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/825_Dims7853_copy_const.npy'.format(ds, height, width)).flatten()))(conv50_12)

conv50_14 = Conv2D(filters=2, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.outconv2.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/828_Dims7709_copy_const.npy'.format(ds, height, width)).flatten()))(conv50_13)
sigm50_2 = sigmoid(conv50_14)
resize50_4 = resize_images(sigm50_2, 4, 4, 'channels_last', interpolation='bilinear')

concat50_5 = Concatenate()([resize19_2, resize50_4])


# Block_51
conv51_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.block4.pre_concat_conv.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/840_Dims7697_copy_const.npy'.format(ds, height, width)).flatten()))(conv50_13)

conv51_2 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.block4.pre_concat_conv.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/843_Dims7673_copy_const.npy'.format(ds, height, width)).flatten()))(conv51_1)
resize51_1 = resize_images(conv51_2, 2, 2, 'channels_last', interpolation='nearest')

concat51_1 = Concatenate()([resize51_1, conv1_1])
# Block_50 ####################################################################################################################


# Block_23
conv23_1 = Conv2D(filters=2, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.outconv3.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/722_Dims7847_copy_const.npy'.format(ds, height, width)).flatten()))(conv22_2)
resize23_1 = resize_images(conv23_1, 2, 2, 'channels_last', interpolation='bilinear')


conv23_2 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.block4.post_concat_conv.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/856_Dims7841_copy_const.npy'.format(ds, height, width)).flatten()))(concat51_1)

conv23_3 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.block4.post_concat_conv.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/859_Dims7727_copy_const.npy'.format(ds, height, width)).flatten()))(conv23_2)

conv23_4 = Conv2D(filters=2, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.outconv3.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/862_Dims7703_copy_const.npy'.format(ds, height, width)).flatten()))(conv23_3)
sigm23_1 = sigmoid(conv23_4)
resize23_2 = resize_images(sigm23_1, 2, 2, 'channels_last', interpolation='bilinear')

concat23_1 = Concatenate()([resize23_1, resize23_2])

# Block_24
resize24_1 = resize_images(conv22_2, 2, 2, 'channels_last', interpolation='nearest')
conv24_1 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.outconv4.0.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/742_Dims7823_copy_const.npy'.format(ds, height, width)).flatten()))(resize24_1)

conv24_2 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.outconv4.0.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/745_Dims7883_copy_const.npy'.format(ds, height, width)).flatten()))(conv24_1)

conv24_3 = Conv2D(filters=2, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_mask_decoder.outconv4.1.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/748_Dims7871_copy_const.npy'.format(ds, height, width)).flatten()))(conv24_2)

# Block_25
resize25_1 = resize_images(conv23_3, 2, 2, 'channels_last', interpolation='nearest')
conv25_1 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.outconv4.0.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/883_Dims7925_copy_const.npy'.format(ds, height, width)).flatten()))(resize25_1)

conv25_2 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.outconv4.0.conv2.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/886_Dims7757_copy_const.npy'.format(ds, height, width)).flatten()))(conv25_1)

conv25_3 = Conv2D(filters=2, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='elu',
                 kernel_initializer=Constant(np.load('weights/{}_{}x{}/FP32/onnx_initializer_node_depth_decoder.outconv4.1.conv1.weight_Output_0_Data__const.npy'.format(ds, height, width)).transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/{}_{}x{}/FP32/889_Dims7745_copy_const.npy'.format(ds, height, width)).flatten()))(conv25_2)

sigm25_1 = sigmoid(conv25_3)

concat25_1 = Concatenate()([conv24_3, sigm25_1])



model = Model(inputs=inputs, outputs=[concat23_1, concat25_1, concat50_3, concat50_5])

model.summary()

tf.saved_model.save(model, 'saved_model_{}_{}x{}'.format(ds, height, width))
# model.save('footprints_{}_{}x{}_float32.h5'.format(ds, height, width).format(height, width))


full_model = tf.function(lambda inputs: model(inputs))
full_model = full_model.get_concrete_function(inputs=[tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)])
frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
frozen_func.graph.as_graph_def()
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=".",
                    name="footprints_{}_{}x{}_float32.pb".format(ds, height, width),
                    as_text=False)


# No Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('footprints_{}_{}x{}_float32.tflite'.format(ds, height, width), 'wb') as w:
    w.write(tflite_model)
print("tflite convert complete! - footprints_{}_{}x{}_float32.tflite".format(ds, height, width))


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
with open('footprints_{}_{}x{}_weight_quant.tflite'.format(ds, height, width), 'wb') as w:
    w.write(tflite_model)
print('Weight Quantization complete! - footprints_{}_{}x{}_weight_quant.tflite'.format(ds, height, width))


def representative_dataset_gen():
    for data in raw_test_data.take(10):
        image = data['image'].numpy()
        image = tf.image.resize(image, (height, width))
        image = image[np.newaxis,:,:,:]
        # image = image / 127.5 - 1.0
        yield [image]

raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)


# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('footprints_{}_{}x{}_integer_quant.tflite'.format(ds, height, width), 'wb') as w:
    w.write(tflite_quant_model)
print('Integer Quantization complete! - footprints_{}_{}x{}_integer_quant.tflite'.format(ds, height, width))


# # Full Integer Quantization - Input/Output=int8
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('footprints_{}_{}x{}_full_integer_quant.tflite'.format(ds, height, width), 'wb') as w:
#     w.write(tflite_quant_model)
# print('Integer Quantization complete! - footprints_{}_{}x{}_full_integer_quant.tflite'.format(ds, height, width))


# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('footprints_{}_{}x{}_float16_quant.tflite'.format(ds, height, width), 'wb') as w:
    w.write(tflite_quant_model)
print('Float16 Quantization complete! - footprints_{}_{}x{}_float16_quant.tflite'.format(ds, height, width))


# # EdgeTPU
# import subprocess
# result = subprocess.check_output(["edgetpu_compiler", "-s", "footprints_{}_{}x{}_full_integer_quant.tflite".format(ds, height, width)])
# print(result)
