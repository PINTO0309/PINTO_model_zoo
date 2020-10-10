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
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose

### https://www.tensorflow.org/api_docs/python/tf/math/multiply
### https://www.tensorflow.org/api_docs/python/tf/linalg/matmul
### https://www.tensorflow.org/api_docs/python/tf/math/sigmoid
### https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/examples/digit_classifier/ml/mnist_tflite.ipynb#scrollTo=2fXStjR4mzkR

### How to initialize a convolution layer with an arbitrary kernel in Keras? https://stackoverrun.com/ja/q/12269118

###  saved_model_cli show --dir saved_model_448x448/ --tag_set serve --signature_def serving_default

'''
he given SavedModel SignatureDef contains the following input(s):
  inputs['input_1'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 448, 448, 3)
      name: serving_default_input_1:0
  inputs['input_4'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 448, 448, 3)
      name: serving_default_input_4:0
  inputs['input_7'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 448, 448, 3)
      name: serving_default_input_7:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['conv2d_41'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 28, 28, 24)
      name: StatefulPartitionedCall:0
  outputs['re_lu_16'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 28, 28, 672)
      name: StatefulPartitionedCall:1
  outputs['tf_op_layer_Tanh'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 28, 28, 48)
      name: StatefulPartitionedCall:2
  outputs['tf_op_layer_Tanh_1'] tensor_info:
      dtype: DT_FLOAT
      shape: (1, 28, 28, 2016)
      name: StatefulPartitionedCall:3
Method name is: tensorflow/serving/predict
'''

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, ReLU, MaxPool2D, Reshape, Concatenate, Conv2DTranspose, Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.backend import resize_images
from tensorflow.keras.activations import tanh
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import sys
import tensorflow_datasets as tfds

# tmp = np.load('weights/448x448/FP32/1426114264_const.npy')
# # tmp = np.load('weights/448x448/FP32/data_add_46134618_copy_const.npy').flatten()#.transpose(0,2,3,1)#.flatten()
# print(tmp.shape)
# print(tmp)
# import sys
# sys.exit(0)

# def init_f(shape, dtype=None):
#        ker = np.load('weights/448x448/FP32/depthwise_conv2d_Kernel')
#        print(shape)
#        return ker

# sys.exit(0)


height = 448
width  = 448

input_1 = Input(shape=(height, width, 3), batch_size=1, name='input_1')
input_4 = Input(shape=(height, width, 3), batch_size=1, name='input_4')
input_7 = Input(shape=(height, width, 3), batch_size=1, name='input_7')



# Block_01
conv1_1 = Conv2D(filters=64, kernel_size=[7, 7], strides=[2, 2], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/341_mean_Fused_Mul_98969898_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_90069011_copy_const.npy').flatten()))(input_1)
maxpool1_1 = tf.nn.max_pool(conv1_1, ksize=[3, 3], strides=[2, 2], padding='SAME')

# Block_02
conv2_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/361_mean_Fused_Mul_99009902_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_90149019_copy_const.npy').flatten()))(maxpool1_1)
conv2_2 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/364_mean_Fused_Mul_99049906_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_90229027_copy_const.npy').flatten()))(conv2_1)
add2_1 = Add()([conv2_2, maxpool1_1])
relu2_1 = ReLU()(add2_1)

# Block_03
conv3_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/368_mean_Fused_Mul_99089910_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_90309035_copy_const.npy').flatten()))(relu2_1)
conv3_2 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/371_mean_Fused_Mul_99129914_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_90389043_copy_const.npy').flatten()))(conv3_1)
add3_1 = Add()([conv3_2, relu2_1])
relu3_1 = ReLU()(add3_1)

# Block_04
conv4_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/375_mean_Fused_Mul_99169918_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_90469051_copy_const.npy').flatten()))(relu3_1)
conv4_2 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/378_mean_Fused_Mul_99209922_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_90549059_copy_const.npy').flatten()))(conv4_1)
add4_1 = Add()([conv4_2, relu3_1])
relu4_1 = ReLU()(add4_1)



# Block_05
conv5_1 = Conv2D(filters=64, kernel_size=[7, 7], strides=[2, 2], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/345_mean_Fused_Mul_98929894_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_89989003_copy_const.npy').flatten()))(input_4)
maxpool5_1 = tf.nn.max_pool(conv5_1, ksize=[3, 3], strides=[2, 2], padding='SAME')

conv5_2 = Conv2D(filters=64, kernel_size=[7, 7], strides=[2, 2], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/349_mean_Fused_Mul_98889890_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_8995_copy_const.npy').flatten()))(input_7)
maxpool5_2 = tf.nn.max_pool(conv5_2, ksize=[3, 3], strides=[2, 2], padding='SAME')

add5_1 = Add()([maxpool5_1, maxpool5_2])

concat5_1 = Concatenate()([add5_1, maxpool1_1])

depthconv5_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1], activation='relu',
                 depthwise_initializer=Constant(np.load('weights/448x448/FP32/1426114264_const.npy').transpose(3,4,1,2,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_90629067_copy_const.npy').flatten()))(concat5_1)
conv5_2 = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/358_mean_Fused_Mul_99289930_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_90709075_copy_const.npy').flatten()))(depthconv5_1)

# Block_06
add6_1 = Add()([conv5_2, relu4_1])


# Block_07
conv7_1 = Conv2D(filters=128, kernel_size=[1, 1], strides=[2, 2], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/388_mean_Fused_Mul_99329934_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_90789083_copy_const.npy').flatten()))(add6_1)

# Block_08
conv8_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[2, 2], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/383_mean_Fused_Mul_99369938_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_90869091_copy_const.npy').flatten()))(add6_1)
conv8_2 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/386_mean_Fused_Mul_99409942_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_90949099_copy_const.npy').flatten()))(conv8_1)
add8_1 = Add()([conv7_1, conv8_2])
relu8_1 = ReLU()(add8_1)

# Block_09
conv9_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/392_mean_Fused_Mul_99449946_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_91029107_copy_const.npy').flatten()))(relu8_1)
conv9_2 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/395_mean_Fused_Mul_99489950_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_91109115_copy_const.npy').flatten()))(conv9_1)
add9_1 = Add()([conv9_2, relu8_1])
relu9_1 = ReLU()(add9_1)

# Block_10
conv10_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/399_mean_Fused_Mul_99529954_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_91189123_copy_const.npy').flatten()))(relu9_1)
conv10_2 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/402_mean_Fused_Mul_99569958_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_91269131_copy_const.npy').flatten()))(conv10_1)
add10_1 = Add()([conv10_2, relu9_1])
relu10_1 = ReLU()(add10_1)

# Block_11
conv11_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/406_mean_Fused_Mul_99609962_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_91349139_copy_const.npy').flatten()))(relu10_1)
conv11_2 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/409_mean_Fused_Mul_99649966_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_91429147_copy_const.npy').flatten()))(conv11_1)
add11_1 = Add()([conv11_2, relu10_1])
relu11_1 = ReLU()(add11_1)

# Block_12
conv12_1 = Conv2D(filters=256, kernel_size=[1, 1], strides=[2, 2], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/418_mean_Fused_Mul_99689970_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_91509155_copy_const.npy').flatten()))(relu11_1)

conv12_2 = Conv2D(filters=256, kernel_size=[3, 3], strides=[2, 2], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/413_mean_Fused_Mul_99729974_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_91589163_copy_const.npy').flatten()))(relu11_1)
conv12_3 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/416_mean_Fused_Mul_99769978_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_91669171_copy_const.npy').flatten()))(conv12_2)

add12_1 = Add()([conv12_1, conv12_3])
relu12_1 = ReLU()(add12_1)

# Block_13
conv13_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/422_mean_Fused_Mul_99809982_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_91749179_copy_const.npy').flatten()))(relu12_1)
conv13_2 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/425_mean_Fused_Mul_99849986_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_91829187_copy_const.npy').flatten()))(conv13_1)

add13_1 = Add()([conv13_2, relu12_1])
relu13_1 = ReLU()(add13_1)

# Block_14
conv14_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/429_mean_Fused_Mul_99889990_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_91909195_copy_const.npy').flatten()))(relu13_1)
conv14_2 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/432_mean_Fused_Mul_99929994_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_91989203_copy_const.npy').flatten()))(conv14_1)

add14_1 = Add()([conv14_2, relu13_1])
relu14_1 = ReLU()(add14_1)

# Block_15
conv15_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/436_mean_Fused_Mul_99969998_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_92069211_copy_const.npy').flatten()))(relu14_1)
conv15_2 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/439_mean_Fused_Mul_1000010002_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_92149219_copy_const.npy').flatten()))(conv15_1)

add15_1 = Add()([conv15_2, relu14_1])
relu15_1 = ReLU()(add15_1)

# Block_16
conv16_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/443_mean_Fused_Mul_1000410006_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_92229227_copy_const.npy').flatten()))(relu15_1)
conv16_2 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/446_mean_Fused_Mul_1000810010_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_92309235_copy_const.npy').flatten()))(conv16_1)

add16_1 = Add()([conv16_2, relu15_1])
relu16_1 = ReLU()(add16_1)

# Block_17
conv17_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/450_mean_Fused_Mul_1001210014_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_92389243_copy_const.npy').flatten()))(relu16_1)
conv17_2 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/453_mean_Fused_Mul_1001610018_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_92469251_copy_const.npy').flatten()))(conv17_1)

add17_1 = Add()([conv17_2, relu16_1])
relu17_1 = ReLU()(add17_1)

# Block_18
conv18_1 = Conv2D(filters=512, kernel_size=[1, 1], strides=[2, 2], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/462_mean_Fused_Mul_1002010022_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_92549259_copy_const.npy').flatten()))(relu17_1)

conv18_2 = Conv2D(filters=512, kernel_size=[3, 3], strides=[2, 2], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/457_mean_Fused_Mul_1002410026_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_92629267_copy_const.npy').flatten()))(relu17_1)
conv18_3 = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/460_mean_Fused_Mul_1002810030_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_92709275_copy_const.npy').flatten()))(conv18_2)

add18_1 = Add()([conv18_1, conv18_3])
relu18_1 = ReLU()(add18_1)

# Block_19
conv19_1 = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/466_mean_Fused_Mul_1003210034_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_92789283_copy_const.npy').flatten()))(relu18_1)
conv19_2 = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/469_mean_Fused_Mul_1003610038_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_92869291_copy_const.npy').flatten()))(conv19_1)

add19_1 = Add()([conv19_2, relu18_1])
relu19_1 = ReLU()(add19_1)

# Block_20
conv20_1 = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/473_mean_Fused_Mul_1004010042_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_92949299_copy_const.npy').flatten()))(relu19_1)
conv20_2 = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/476_mean_Fused_Mul_1004410046_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_93029307_copy_const.npy').flatten()))(conv20_1)

add20_1 = Add()([conv20_2, relu19_1])
relu20_1 = ReLU()(add20_1)

# Block_21
depthconv21_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1], activation='relu',
                 depthwise_initializer=Constant(np.load('weights/448x448/FP32/1428914292_const.npy').transpose(3,4,1,2,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_93109315_copy_const.npy').flatten()))(relu20_1)
conv21_1 = Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/483_mean_Fused_Mul_1005210054_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_93189323_copy_const.npy').flatten()))(depthconv21_1)
conv2dtran21_1 = Conv2DTranspose(filters=512, kernel_size=[2, 2], strides=[2, 2], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/486_mean_Fused_Mul_1005610058_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_93269331_copy_const.npy').flatten()))(conv21_1)
depthconv21_2 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1], activation='relu',
                 depthwise_initializer=Constant(np.load('weights/448x448/FP32/1429314296_const.npy').transpose(3,4,1,2,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_93349339_copy_const.npy').flatten()))(conv2dtran21_1)
conv21_2 = Conv2D(filters=1024, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/492_mean_Fused_Mul_1006410066_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_93429347_copy_const.npy').flatten()))(depthconv21_2)

# Block_22
depthconv22_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1], activation='relu',
                 depthwise_initializer=Constant(np.load('weights/448x448/FP32/1427714280_const.npy').transpose(3,4,1,2,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_94229427_copy_const.npy').flatten()))(conv21_2)
conv22_1 = Conv2D(filters=24, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/498_mean_Fused_Mul_1010010102_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_94309435_copy_const.npy').flatten()))(depthconv22_1)

# Block_23
depthconv23_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1], activation='relu',
                 depthwise_initializer=Constant(np.load('weights/448x448/FP32/1427314276_const.npy').transpose(3,4,1,2,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_94149419_copy_const.npy').flatten()))(conv21_2)
conv23_1 = Conv2D(filters=48, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/onnx_initializer_node_offset.3.weight_Output_0_Data__const.npy').transpose(2,3,1,0)))(depthconv23_1)
tanh23_1 = tanh(conv23_1)

# Block_24
depthconv24_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1], activation='relu',
                 depthwise_initializer=Constant(np.load('weights/448x448/FP32/1426514268_const.npy').transpose(3,4,1,2,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_93509355_copy_const.npy').flatten()))(conv21_2)
conv24_1 = Conv2D(filters=1024, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/509_mean_Fused_Mul_1007210074_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_93589363_copy_const.npy').flatten()))(depthconv24_1)

# Block_25
depthconv25_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1], activation='relu',
                 depthwise_initializer=Constant(np.load('weights/448x448/FP32/1426914272_const.npy').transpose(3,4,1,2,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_93829387_copy_const.npy').flatten()))(conv24_1)
conv25_1 = Conv2D(filters=672, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/onnx_initializer_node_model3_0.3.weight_Output_0_Data__const.npy').transpose(2,3,1,0)))(depthconv25_1)

mul25_1 = tf.math.multiply(conv25_1, np.load('weights/448x448/FP32/data_mul_94059409_copy_const.npy').transpose(0,2,3,1))
add25_1 = Add()([mul25_1, np.load('weights/448x448/FP32/data_add_94069411_copy_const.npy').transpose(0,2,3,1)])
relu25_1 = ReLU()(add25_1)

mul25_2 = tf.math.multiply(conv25_1, np.load('weights/448x448/FP32/data_mul_93899393_copy_const.npy').transpose(0,2,3,1))
add25_2 = Add()([mul25_2, np.load('weights/448x448/FP32/data_add_93909395_copy_const.npy').transpose(0,2,3,1)])
relu25_2 = ReLU()(add25_2)

# Block_26
depthconv26_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1], activation='relu',
                 depthwise_initializer=Constant(np.load('weights/448x448/FP32/1428514288_const.npy').transpose(3,4,1,2,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_93669371_copy_const.npy').flatten()))(conv24_1)
conv26_1 = Conv2D(filters=672, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/521_mean_Fused_Mul_1008010082_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_93749379_copy_const.npy').flatten()))(depthconv26_1)
concat26_1 = Concatenate()([relu25_2, conv26_1])

# Block_27
depthconv27_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=1, dilation_rate=[1, 1], activation='relu',
                 depthwise_initializer=Constant(np.load('weights/448x448/FP32/1428114284_const.npy').transpose(3,4,1,2,0)),
                 bias_initializer=Constant(np.load('weights/448x448/FP32/data_add_93989403_copy_const.npy').flatten()))(concat26_1)
conv27_1 = Conv2D(filters=2016, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/448x448/FP32/onnx_initializer_node_offset3D.3.weight_Output_0_Data__const.npy').transpose(2,3,1,0)))(depthconv27_1)
tanh27_1 = tanh(conv27_1)



model = Model(inputs=[input_1, input_4, input_7], outputs=[conv22_1, tanh23_1, relu25_1, tanh27_1])

model.summary()

# tf.saved_model.save(model, 'saved_model_{}x{}'.format(height, width))
# model.save('resnet34_3inputs_{}x{}_float32.h5'.format(height, width))

# full_model = tf.function(lambda inputs: model(inputs))
# full_model = full_model.get_concrete_function(inputs=[tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype),
#                                                       tf.TensorSpec(model.inputs[1].shape, model.inputs[1].dtype),
#                                                       tf.TensorSpec(model.inputs[2].shape, model.inputs[2].dtype)])
# frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
# frozen_func.graph.as_graph_def()
# tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
#                     logdir=".",
#                     name="resnet34_3inputs_{}x{}_float32.pb".format(height, width),
#                     as_text=False)

# No Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('resnet34_3inputs_{}x{}_float32.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print("tflite convert complete! - resnet34_3inputs_{}x{}_float32.tflite".format(height, width))


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
with open('resnet34_3inputs_{}x{}_weight_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Weight Quantization complete! - resnet34_3inputs_{}x{}_weight_quant.tflite'.format(height, width))


def representative_dataset_gen():
    for data in raw_test_data.take(10):
        image = data['image'].numpy()
        image = tf.image.resize(image, (height, width))
        image = image[np.newaxis,:,:,:]
        image = image / 127.5 - 1.0
        yield [image, image, image]

raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)


# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('resnet34_3inputs_{}x{}_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_quant_model)
print('Integer Quantization complete! - resnet34_3inputs_{}x{}_integer_quant.tflite'.format(height, width))


# Full Integer Quantization - Input/Output=int8
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('resnet34_3inputs_{}x{}_full_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_quant_model)
print('Integer Quantization complete! - resnet34_3inputs_{}x{}_full_integer_quant.tflite'.format(height, width))


# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('resnet34_3inputs_{}x{}_float16_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_quant_model)
print('Float16 Quantization complete! - resnet34_3inputs_{}x{}_float16_quant.tflite'.format(height, width))


# EdgeTPU
import subprocess
result = subprocess.check_output(["edgetpu_compiler", "-s", "resnet34_3inputs_{}x{}_full_integer_quant.tflite".format(height, width)])
print(result)
