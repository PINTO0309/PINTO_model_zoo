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

### https://www.tensorflow.org/api_docs/python/tf/math/multiply
### https://www.tensorflow.org/api_docs/python/tf/linalg/matmul
### https://www.tensorflow.org/api_docs/python/tf/math/sigmoid
### https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/examples/digit_classifier/ml/mnist_tflite.ipynb#scrollTo=2fXStjR4mzkR

### How to initialize a convolution layer with an arbitrary kernel in Keras? https://stackoverrun.com/ja/q/12269118

###  saved_model_cli show --dir saved_model_pose_detection/ --tag_set serve --signature_def serving_default

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, ReLU, MaxPool2D, Reshape, Concatenate, Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.backend import resize_images
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import sys
import tensorflow_datasets as tfds

# tmp = np.load('weights_u2netp/480x640/FP32/depthwise_conv2d_Kernel')
# print(tmp.shape)
# print(tmp)

# def init_f(shape, dtype=None):
#        ker = np.load('weights_u2netp/480x640/FP32/depthwise_conv2d_Kernel')
#        print(shape)
#        return ker

# sys.exit(0)


height = 320
width  = 320
inputs = Input(shape=(height, width, 3), batch_size=1, name='input')

# Block_01
conv1_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/800_mean_Fused_Mul_2756627568_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_25668_copy_const.npy').flatten()))(inputs)
relu1_1 = ReLU()(conv1_1)

# Block_02
conv2_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/803_mean_Fused_Mul_2757027572_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2567125676_copy_const.npy').flatten()))(relu1_1)
relu2_1 = ReLU()(conv2_1)

# Block_03
maxpool3_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu2_1)
conv3_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/807_mean_Fused_Mul_2757427576_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2567925684_copy_const.npy').flatten()))(maxpool3_1)
relu3_1 = ReLU()(conv3_1)

# Block_04
maxpool4_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu3_1)
conv4_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/811_mean_Fused_Mul_2757827580_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2568725692_copy_const.npy').flatten()))(maxpool4_1)
relu4_1 = ReLU()(conv4_1)

# Block_05
maxpool5_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu4_1)
conv5_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/815_mean_Fused_Mul_2758227584_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2569525700_copy_const.npy').flatten()))(maxpool5_1)
relu5_1 = ReLU()(conv5_1)

# Block_06
maxpool6_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu5_1)
conv6_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/819_mean_Fused_Mul_2758627588_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2570325708_copy_const.npy').flatten()))(maxpool6_1)
relu6_1 = ReLU()(conv6_1)

# Block_07
maxpool7_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu6_1)
conv7_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/823_mean_Fused_Mul_2759027592_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2571125716_copy_const.npy').flatten()))(maxpool7_1)
relu7_1 = ReLU()(conv7_1)

# Block_08
conv8_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[2, 2],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/826_mean_Fused_Mul_2759427596_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2571925724_copy_const.npy').flatten()))(relu7_1)
relu8_1 = ReLU()(conv8_1)
concat8_1 = Concatenate(axis=-1)([relu7_1, relu8_1])

# Block_09
conv9_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/830_mean_Fused_Mul_2759827600_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2572725732_copy_const.npy').flatten()))(concat8_1)
relu9_1 = ReLU()(conv9_1)
resize9_1 = resize_images(relu9_1, 2, 2, 'channels_last', interpolation='bilinear')
concat9_1 = Concatenate(axis=-1)([relu6_1, resize9_1])

# Block_10
conv10_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/851_mean_Fused_Mul_2760227604_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2573525740_copy_const.npy').flatten()))(concat9_1)
relu10_1 = ReLU()(conv10_1)
resize10_1 = resize_images(relu10_1, 2, 2, 'channels_last', interpolation='bilinear')
concat10_1 = Concatenate(axis=-1)([relu5_1, resize10_1])

# Block_11
conv11_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/872_mean_Fused_Mul_2760627608_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2574325748_copy_const.npy').flatten()))(concat10_1)
relu11_1 = ReLU()(conv11_1)
resize11_1 = resize_images(relu11_1, 2, 2, 'channels_last', interpolation='bilinear')
concat11_1 = Concatenate(axis=-1)([relu4_1, resize11_1])

# Block_12
conv12_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/893_mean_Fused_Mul_2761027612_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2575125756_copy_const.npy').flatten()))(concat11_1)
relu12_1 = ReLU()(conv12_1)
resize12_1 = resize_images(relu12_1, 2, 2, 'channels_last', interpolation='bilinear')
concat12_1 = Concatenate(axis=-1)([relu3_1, resize12_1])

# Block_13
conv13_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/914_mean_Fused_Mul_2761427616_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2575925764_copy_const.npy').flatten()))(concat12_1)
relu13_1 = ReLU()(conv13_1)
resize13_1 = resize_images(relu13_1, 2, 2, 'channels_last', interpolation='bilinear')
concat13_1 = Concatenate(axis=-1)([relu2_1, resize13_1])

# Block_14
conv14_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/935_mean_Fused_Mul_2761827620_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2576725772_copy_const.npy').flatten()))(concat13_1)
relu14_1 = ReLU()(conv14_1)
add14_2 = Add()([relu1_1, relu14_1])



# Block_15
maxpool15_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(add14_2)
conv15_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/940_mean_Fused_Mul_2762227624_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2577525780_copy_const.npy').flatten()))(maxpool15_1)
relu15_1 = ReLU()(conv15_1)

# Block_16
conv16_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/943_mean_Fused_Mul_2762627628_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2578325788_copy_const.npy').flatten()))(relu15_1)
relu16_1 = ReLU()(conv16_1)

# Block_17
maxpool17_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu16_1)
conv17_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/947_mean_Fused_Mul_2763027632_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2579125796_copy_const.npy').flatten()))(maxpool17_1)
relu17_1 = ReLU()(conv17_1)

# Block_18
maxpool18_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu17_1)
conv18_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/951_mean_Fused_Mul_2763427636_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2579925804_copy_const.npy').flatten()))(maxpool18_1)
relu18_1 = ReLU()(conv18_1)

# Block_19
maxpool19_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu18_1)
conv19_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/955_mean_Fused_Mul_2763827640_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2580725812_copy_const.npy').flatten()))(maxpool19_1)
relu19_1 = ReLU()(conv19_1)

# Block_20
maxpool20_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu19_1)
conv20_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/959_mean_Fused_Mul_2764227644_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2581525820_copy_const.npy').flatten()))(maxpool20_1)
relu20_1 = ReLU()(conv20_1)

# Block_21
conv21_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[2, 2],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/962_mean_Fused_Mul_2764627648_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2582325828_copy_const.npy').flatten()))(relu20_1)
relu21_1 = ReLU()(conv21_1)
concat21_1 = Concatenate(axis=-1)([relu20_1, relu21_1])

# Block_22
conv22_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/966_mean_Fused_Mul_2765027652_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2583125836_copy_const.npy').flatten()))(concat21_1)
relu22_1 = ReLU()(conv22_1)
resize22_1 = resize_images(relu22_1, 2, 2, 'channels_last', interpolation='bilinear')
concat22_1 = Concatenate(axis=-1)([relu19_1, resize22_1])

# Block_23
conv23_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/987_mean_Fused_Mul_2765427656_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2583925844_copy_const.npy').flatten()))(concat22_1)
relu23_1 = ReLU()(conv23_1)
resize23_1 = resize_images(relu23_1, 2, 2, 'channels_last', interpolation='bilinear')
concat23_1 = Concatenate(axis=-1)([relu18_1, resize23_1])

# Block_24
conv24_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1008_mean_Fused_Mul_2765827660_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2584725852_copy_const.npy').flatten()))(concat23_1)
relu24_1 = ReLU()(conv24_1)
resize24_1 = resize_images(relu24_1, 2, 2, 'channels_last', interpolation='bilinear')
concat24_1 = Concatenate(axis=-1)([relu17_1, resize24_1])

# Block_25
conv25_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1029_mean_Fused_Mul_2766227664_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2585525860_copy_const.npy').flatten()))(concat24_1)
relu25_1 = ReLU()(conv25_1)
resize25_1 = resize_images(relu25_1, 2, 2, 'channels_last', interpolation='bilinear')
concat25_1 = Concatenate(axis=-1)([relu16_1, resize25_1])



# Block_26
conv26_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1050_mean_Fused_Mul_2766627668_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2586325868_copy_const.npy').flatten()))(concat25_1)
relu26_1 = ReLU()(conv26_1)
add26_2 = Add()([relu15_1, relu26_1])

# Block_27
maxpool27_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(add26_2)
conv27_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1055_mean_Fused_Mul_2767027672_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2587125876_copy_const.npy').flatten()))(maxpool27_1)
relu27_1 = ReLU()(conv27_1)

# Block_28
conv28_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1058_mean_Fused_Mul_2767427676_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2587925884_copy_const.npy').flatten()))(relu27_1)
relu28_1 = ReLU()(conv28_1)

# Block_29
maxpool29_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu28_1)
conv29_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1062_mean_Fused_Mul_2767827680_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2588725892_copy_const.npy').flatten()))(maxpool29_1)
relu29_1 = ReLU()(conv29_1)

# Block_30
maxpool30_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu29_1)
conv30_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1066_mean_Fused_Mul_2768227684_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2589525900_copy_const.npy').flatten()))(maxpool30_1)
relu30_1 = ReLU()(conv30_1)

# Block_31
maxpool31_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu30_1)
conv31_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1070_mean_Fused_Mul_2768627688_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2590325908_copy_const.npy').flatten()))(maxpool31_1)
relu31_1 = ReLU()(conv31_1)

# Block_32
conv32_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[2, 2],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1073_mean_Fused_Mul_2769027692_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2591125916_copy_const.npy').flatten()))(relu31_1)
relu32_1 = ReLU()(conv32_1)
concat32_1 = Concatenate(axis=-1)([relu31_1, relu32_1])

# Block_33
conv33_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1077_mean_Fused_Mul_2769427696_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2591925924_copy_const.npy').flatten()))(concat32_1)
relu33_1 = ReLU()(conv33_1)
resize33_1 = resize_images(relu33_1, 2, 2, 'channels_last', interpolation='bilinear')
concat33_1 = Concatenate(axis=-1)([relu30_1, resize33_1])

# Block_34
conv34_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1098_mean_Fused_Mul_2769827700_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2592725932_copy_const.npy').flatten()))(concat33_1)
relu34_1 = ReLU()(conv34_1)
resize34_1 = resize_images(relu34_1, 2, 2, 'channels_last', interpolation='bilinear')
concat34_1 = Concatenate(axis=-1)([relu29_1, resize34_1])

# Block_35
conv35_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1119_mean_Fused_Mul_2770227704_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2593525940_copy_const.npy').flatten()))(concat34_1)
relu35_1 = ReLU()(conv35_1)
resize35_1 = resize_images(relu35_1, 2, 2, 'channels_last', interpolation='bilinear')
concat35_1 = Concatenate(axis=-1)([relu28_1, resize35_1])

# Block_36
conv36_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1140_mean_Fused_Mul_2770627708_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2594325948_copy_const.npy').flatten()))(concat35_1)
relu36_1 = ReLU()(conv36_1)
add36_2 = Add()([relu27_1, relu36_1])



# Block_37
maxpool37_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(add36_2)
conv37_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1145_mean_Fused_Mul_2771027712_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2595125956_copy_const.npy').flatten()))(maxpool37_1)
relu37_1 = ReLU()(conv37_1)

# Block_38
conv38_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1148_mean_Fused_Mul_2771427716_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2595925964_copy_const.npy').flatten()))(relu37_1)
relu38_1 = ReLU()(conv38_1)

# Block_39
maxpool39_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu38_1)
conv39_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1152_mean_Fused_Mul_2771827720_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2596725972_copy_const.npy').flatten()))(maxpool39_1)
relu39_1 = ReLU()(conv39_1)

# Block_40
maxpool40_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu39_1)
conv40_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1156_mean_Fused_Mul_2772227724_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2597525980_copy_const.npy').flatten()))(maxpool40_1)
relu40_1 = ReLU()(conv40_1)

# Block_41
conv41_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[2, 2],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1159_mean_Fused_Mul_2772627728_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2598325988_copy_const.npy').flatten()))(relu40_1)
relu41_1 = ReLU()(conv41_1)
concat41_1 = Concatenate(axis=-1)([relu40_1, relu41_1])

# Block_42
conv42_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1163_mean_Fused_Mul_2773027732_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2599125996_copy_const.npy').flatten()))(concat41_1)
relu42_1 = ReLU()(conv42_1)
resize42_1 = resize_images(relu42_1, 2, 2, 'channels_last', interpolation='bilinear')
concat42_1 = Concatenate(axis=-1)([relu39_1, resize42_1])

# Block_43
conv43_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1184_mean_Fused_Mul_2773427736_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2599926004_copy_const.npy').flatten()))(concat42_1)
relu43_1 = ReLU()(conv43_1)
resize43_1 = resize_images(relu43_1, 2, 2, 'channels_last', interpolation='bilinear')
concat43_1 = Concatenate(axis=-1)([relu38_1, resize43_1])

# Block_44
conv44_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1205_mean_Fused_Mul_2773827740_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2600726012_copy_const.npy').flatten()))(concat43_1)
relu44_1 = ReLU()(conv44_1)
add44_2 = Add()([relu37_1, relu44_1])



# Block_45
maxpool45_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(add44_2)
conv45_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1210_mean_Fused_Mul_2774227744_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2601526020_copy_const.npy').flatten()))(maxpool45_1)
relu45_1 = ReLU()(conv45_1)

# Block_46
conv46_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1213_mean_Fused_Mul_2774627748_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2602326028_copy_const.npy').flatten()))(relu45_1)
relu46_1 = ReLU()(conv46_1)

# Block_47
conv47_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[2, 2],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1216_mean_Fused_Mul_2775027752_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2603126036_copy_const.npy').flatten()))(relu46_1)
relu47_1 = ReLU()(conv47_1)

# Block_48
conv48_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[4, 4],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1219_mean_Fused_Mul_2775427756_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2603926044_copy_const.npy').flatten()))(relu47_1)
relu48_1 = ReLU()(conv48_1)

# Block_49
conv49_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[8, 8],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1222_mean_Fused_Mul_2775827760_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2604726052_copy_const.npy').flatten()))(relu48_1)
relu49_1 = ReLU()(conv49_1)
concat49_1 = Concatenate(axis=-1)([relu48_1, relu49_1])

# Block_50
conv50_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[4, 4],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1226_mean_Fused_Mul_2776227764_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2605526060_copy_const.npy').flatten()))(concat49_1)
relu50_1 = ReLU()(conv50_1)
concat50_1 = Concatenate(axis=-1)([relu47_1, relu50_1])

# Block_51
conv51_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[2, 2],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1230_mean_Fused_Mul_2776627768_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2606326068_copy_const.npy').flatten()))(concat50_1)
relu51_1 = ReLU()(conv51_1)
concat51_1 = Concatenate(axis=-1)([relu46_1, relu51_1])

# Block_52
conv52_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1234_mean_Fused_Mul_2777027772_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2607126076_copy_const.npy').flatten()))(concat51_1)
relu52_1 = ReLU()(conv52_1)
add52_2 = Add()([relu45_1, relu52_1])



# Block_53
maxpool53_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(add52_2)
conv53_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1239_mean_Fused_Mul_2777427776_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2607926084_copy_const.npy').flatten()))(maxpool53_1)
relu53_1 = ReLU()(conv53_1)

# Block_54
conv54_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1242_mean_Fused_Mul_2777827780_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2608726092_copy_const.npy').flatten()))(relu53_1)
relu54_1 = ReLU()(conv54_1)

# Block_55
conv55_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[2, 2],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1245_mean_Fused_Mul_2778227784_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2609526100_copy_const.npy').flatten()))(relu54_1)
relu55_1 = ReLU()(conv55_1)

# Block_56
conv56_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[4, 4],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1248_mean_Fused_Mul_2778627788_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2610326108_copy_const.npy').flatten()))(relu55_1)
relu56_1 = ReLU()(conv56_1)

# Block_57
conv57_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[8, 8],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1251_mean_Fused_Mul_2779027792_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2611126116_copy_const.npy').flatten()))(relu56_1)
relu57_1 = ReLU()(conv57_1)
concat57_1 = Concatenate(axis=-1)([relu56_1, relu57_1])

# Block_58
conv58_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[4, 4],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1255_mean_Fused_Mul_2779427796_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2611926124_copy_const.npy').flatten()))(concat57_1)
relu58_1 = ReLU()(conv58_1)
concat58_1 = Concatenate(axis=-1)([relu55_1, relu58_1])

# Block_59
conv59_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[2, 2],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1259_mean_Fused_Mul_2779827800_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2612726132_copy_const.npy').flatten()))(concat58_1)
relu59_1 = ReLU()(conv59_1)
concat59_1 = Concatenate(axis=-1)([relu54_1, relu59_1])

# Block_60
conv60_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1263_mean_Fused_Mul_2780227804_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2613526140_copy_const.npy').flatten()))(concat59_1)
relu60_1 = ReLU()(conv60_1)
add60_2 = Add()([relu53_1, relu60_1])
resize60_1 = resize_images(add60_2, 2, 2, 'channels_last', interpolation='bilinear')
concat60_1 = Concatenate(axis=-1)([add52_2, resize60_1])
conv60_2 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1285_mean_Fused_Mul_2780627808_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2614326148_copy_const.npy').flatten()))(concat60_1)
relu60_2 = ReLU()(conv60_2)

# Block_61
conv61_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1288_mean_Fused_Mul_2781027812_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2615126156_copy_const.npy').flatten()))(relu60_2)
relu61_1 = ReLU()(conv61_1)

# Block_62
conv62_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[2, 2],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1291_mean_Fused_Mul_2781427816_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2615926164_copy_const.npy').flatten()))(relu61_1)
relu62_1 = ReLU()(conv62_1)

# Block_63
conv63_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[4, 4],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1294_mean_Fused_Mul_2781827820_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2616726172_copy_const.npy').flatten()))(relu62_1)
relu63_1 = ReLU()(conv63_1)

# Block_64
conv64_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[8, 8],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1297_mean_Fused_Mul_2782227824_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2617526180_copy_const.npy').flatten()))(relu63_1)
relu64_1 = ReLU()(conv64_1)
concat64_1 = Concatenate(axis=-1)([relu63_1, relu64_1])

# Block_65
conv65_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[4, 4],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1301_mean_Fused_Mul_2782627828_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2618326188_copy_const.npy').flatten()))(concat64_1)
relu65_1 = ReLU()(conv65_1)
concat65_1 = Concatenate(axis=-1)([relu62_1, relu65_1])

# Block_66
conv66_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[2, 2],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1305_mean_Fused_Mul_2783027832_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2619126196_copy_const.npy').flatten()))(concat65_1)
relu66_1 = ReLU()(conv66_1)
concat66_1 = Concatenate(axis=-1)([relu61_1, relu66_1])

# Block_67
conv67_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1309_mean_Fused_Mul_2783427836_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2619926204_copy_const.npy').flatten()))(concat66_1)
relu67_1 = ReLU()(conv67_1)
add67_2 = Add()([relu60_2, relu67_1])
resize67_1 = resize_images(add67_2, 2, 2, 'channels_last', interpolation='bilinear')
concat67_1 = Concatenate(axis=-1)([add44_2, resize67_1])

# Block_68
conv68_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1331_mean_Fused_Mul_2783827840_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2620726212_copy_const.npy').flatten()))(concat67_1)
relu68_1 = ReLU()(conv68_1)

# Block_69
conv69_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1334_mean_Fused_Mul_2784227844_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2621526220_copy_const.npy').flatten()))(relu68_1)
relu69_1 = ReLU()(conv69_1)

# Block_70
maxpool70_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu69_1)
conv70_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1338_mean_Fused_Mul_2784627848_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2622326228_copy_const.npy').flatten()))(maxpool70_1)
relu70_1 = ReLU()(conv70_1)

# Block_71
maxpool71_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu70_1)
conv71_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1342_mean_Fused_Mul_2785027852_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2623126236_copy_const.npy').flatten()))(maxpool71_1)
relu71_1 = ReLU()(conv71_1)

# Block_72
conv72_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[2, 2],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1345_mean_Fused_Mul_2785427856_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2623926244_copy_const.npy').flatten()))(relu71_1)
relu72_1 = ReLU()(conv72_1)
concat72_1 = Concatenate(axis=-1)([relu71_1, relu72_1])

# Block_73
conv73_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1349_mean_Fused_Mul_2785827860_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2624726252_copy_const.npy').flatten()))(concat72_1)
relu73_1 = ReLU()(conv73_1)
resize73_1 = resize_images(relu73_1, 2, 2, 'channels_last', interpolation='bilinear')
concat73_1 = Concatenate(axis=-1)([relu70_1, resize73_1])

# Block_74
conv74_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1370_mean_Fused_Mul_2786227864_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2625526260_copy_const.npy').flatten()))(concat73_1)
relu74_1 = ReLU()(conv74_1)
resize74_1 = resize_images(relu74_1, 2, 2, 'channels_last', interpolation='bilinear')
concat74_1 = Concatenate(axis=-1)([relu69_1, resize74_1])

# Block_75
conv75_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1391_mean_Fused_Mul_2786627868_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2626326268_copy_const.npy').flatten()))(concat74_1)
relu75_1 = ReLU()(conv75_1)
add75_2 = Add()([relu68_1, relu75_1])
resize75_1 = resize_images(add75_2, 2, 2, 'channels_last', interpolation='bilinear')
concat75_1 = Concatenate(axis=-1)([add36_2, resize75_1])

# Block_76
conv76_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1413_mean_Fused_Mul_2787027872_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2627126276_copy_const.npy').flatten()))(concat75_1)
relu76_1 = ReLU()(conv76_1)

# Block_77
conv77_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1416_mean_Fused_Mul_2787427876_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2627926284_copy_const.npy').flatten()))(relu76_1)
relu77_1 = ReLU()(conv77_1)

# Block_78
maxpool78_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu77_1)
conv78_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1420_mean_Fused_Mul_2787827880_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2628726292_copy_const.npy').flatten()))(maxpool78_1)
relu78_1 = ReLU()(conv78_1)

# Block_79
maxpool79_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu78_1)
conv79_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1424_mean_Fused_Mul_2788227884_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2629526300_copy_const.npy').flatten()))(maxpool79_1)
relu79_1 = ReLU()(conv79_1)

# Block_80
maxpool80_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu79_1)
conv80_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1428_mean_Fused_Mul_2788627888_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2630326308_copy_const.npy').flatten()))(maxpool80_1)
relu80_1 = ReLU()(conv80_1)

# Block_81
conv81_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[2, 2],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1431_mean_Fused_Mul_2789027892_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2631126316_copy_const.npy').flatten()))(relu80_1)
relu81_1 = ReLU()(conv81_1)
concat81_1 = Concatenate(axis=-1)([relu80_1, relu81_1])

# Block_82
conv82_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1435_mean_Fused_Mul_2789427896_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2631926324_copy_const.npy').flatten()))(concat81_1)
relu82_1 = ReLU()(conv82_1)
resize82_1 = resize_images(relu82_1, 2, 2, 'channels_last', interpolation='bilinear')
concat82_1 = Concatenate(axis=-1)([relu79_1, resize82_1])

# Block_83
conv83_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1456_mean_Fused_Mul_2789827900_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2632726332_copy_const.npy').flatten()))(concat82_1)
relu83_1 = ReLU()(conv83_1)
resize83_1 = resize_images(relu83_1, 2, 2, 'channels_last', interpolation='bilinear')
concat83_1 = Concatenate(axis=-1)([relu78_1, resize83_1])

# Block_84
conv84_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1477_mean_Fused_Mul_2790227904_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2633526340_copy_const.npy').flatten()))(concat83_1)
relu84_1 = ReLU()(conv84_1)
resize84_1 = resize_images(relu84_1, 2, 2, 'channels_last', interpolation='bilinear')
concat84_1 = Concatenate(axis=-1)([relu77_1, resize84_1])

# Block_85
conv85_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1498_mean_Fused_Mul_2790627908_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2634326348_copy_const.npy').flatten()))(concat84_1)
relu85_1 = ReLU()(conv85_1)
add85_2 = Add()([relu76_1, relu85_1])
resize85_1 = resize_images(add85_2, 2, 2, 'channels_last', interpolation='bilinear')
concat85_1 = Concatenate(axis=-1)([add26_2, resize85_1])

# Block_86
conv86_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1520_mean_Fused_Mul_2791027912_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2635126356_copy_const.npy').flatten()))(concat85_1)
relu86_1 = ReLU()(conv86_1)

# Block_87
conv87_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1523_mean_Fused_Mul_2791427916_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2635926364_copy_const.npy').flatten()))(relu86_1)
relu87_1 = ReLU()(conv87_1)

# Block_88
maxpool88_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu87_1)
conv88_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1527_mean_Fused_Mul_2791827920_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2636726372_copy_const.npy').flatten()))(maxpool88_1)
relu88_1 = ReLU()(conv88_1)

# Block_89
maxpool89_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu88_1)
conv89_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1531_mean_Fused_Mul_2792227924_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2637526380_copy_const.npy').flatten()))(maxpool89_1)
relu89_1 = ReLU()(conv89_1)

# Block_90
maxpool90_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu89_1)
conv90_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1535_mean_Fused_Mul_2792627928_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2638326388_copy_const.npy').flatten()))(maxpool90_1)
relu90_1 = ReLU()(conv90_1)

# Block_91
maxpool91_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu90_1)
conv91_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1539_mean_Fused_Mul_2793027932_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2639126396_copy_const.npy').flatten()))(maxpool91_1)
relu91_1 = ReLU()(conv91_1)

# Block_92
conv92_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[2, 2],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1542_mean_Fused_Mul_2793427936_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2639926404_copy_const.npy').flatten()))(relu91_1)
relu92_1 = ReLU()(conv92_1)
concat92_1 = Concatenate(axis=-1)([relu91_1, relu92_1])

# Block_93
conv93_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1546_mean_Fused_Mul_2793827940_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2640726412_copy_const.npy').flatten()))(concat92_1)
relu93_1 = ReLU()(conv93_1)
resize93_1 = resize_images(relu93_1, 2, 2, 'channels_last', interpolation='bilinear')
concat93_1 = Concatenate(axis=-1)([relu90_1, resize93_1])

# Block_94
conv94_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1567_mean_Fused_Mul_2794227944_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2641526420_copy_const.npy').flatten()))(concat93_1)
relu94_1 = ReLU()(conv94_1)
resize94_1 = resize_images(relu94_1, 2, 2, 'channels_last', interpolation='bilinear')
concat94_1 = Concatenate(axis=-1)([relu89_1, resize94_1])

# Block_95
conv95_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1588_mean_Fused_Mul_2794627948_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2642326428_copy_const.npy').flatten()))(concat94_1)
relu95_1 = ReLU()(conv95_1)
resize95_1 = resize_images(relu95_1, 2, 2, 'channels_last', interpolation='bilinear')
concat95_1 = Concatenate(axis=-1)([relu88_1, resize95_1])

# Block_96
conv96_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1609_mean_Fused_Mul_2795027952_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2643126436_copy_const.npy').flatten()))(concat95_1)
relu96_1 = ReLU()(conv96_1)
resize96_1 = resize_images(relu96_1, 2, 2, 'channels_last', interpolation='bilinear')
concat96_1 = Concatenate(axis=-1)([relu87_1, resize96_1])

# Block_97
conv97_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1630_mean_Fused_Mul_2795427956_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2643926444_copy_const.npy').flatten()))(concat96_1)
relu97_1 = ReLU()(conv97_1)
add97_2 = Add()([relu86_1, relu97_1])
resize97_1 = resize_images(add97_2, 2, 2, 'channels_last', interpolation='bilinear')
concat97_1 = Concatenate(axis=-1)([add14_2, resize97_1])

# Block_98
conv98_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1652_mean_Fused_Mul_2795827960_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2644726452_copy_const.npy').flatten()))(concat97_1)
relu98_1 = ReLU()(conv98_1)

# Block_99
conv99_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1655_mean_Fused_Mul_2796227964_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2645526460_copy_const.npy').flatten()))(relu98_1)
relu99_1 = ReLU()(conv99_1)

# Block_100
maxpool100_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu99_1)
conv100_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1659_mean_Fused_Mul_2796627968_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2646326468_copy_const.npy').flatten()))(maxpool100_1)
relu100_1 = ReLU()(conv100_1)

# Block_101
maxpool101_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu100_1)
conv101_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1663_mean_Fused_Mul_2797027972_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2647126476_copy_const.npy').flatten()))(maxpool101_1)
relu101_1 = ReLU()(conv101_1)

# Block_102
maxpool102_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu101_1)
conv102_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1667_mean_Fused_Mul_2797427976_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2647926484_copy_const.npy').flatten()))(maxpool102_1)
relu102_1 = ReLU()(conv102_1)

# Block_103
maxpool103_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu102_1)
conv103_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1671_mean_Fused_Mul_2797827980_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2648726492_copy_const.npy').flatten()))(maxpool103_1)
relu103_1 = ReLU()(conv103_1)

# Block_104
maxpool104_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu103_1)
conv104_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1675_mean_Fused_Mul_2798227984_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2649526500_copy_const.npy').flatten()))(maxpool104_1)
relu104_1 = ReLU()(conv104_1)

# Block_105
conv105_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[2, 2],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1678_mean_Fused_Mul_2798627988_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2650326508_copy_const.npy').flatten()))(relu104_1)
relu105_1 = ReLU()(conv105_1)
concat105_1 = Concatenate(axis=-1)([relu104_1, relu105_1])

# Block_106
conv106_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1682_mean_Fused_Mul_2799027992_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2651126516_copy_const.npy').flatten()))(concat105_1)
relu106_1 = ReLU()(conv106_1)
resize106_1 = resize_images(relu106_1, 2, 2, 'channels_last', interpolation='bilinear')
concat106_1 = Concatenate(axis=-1)([relu103_1, resize106_1])

# Block_107
conv107_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1703_mean_Fused_Mul_2799427996_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2651926524_copy_const.npy').flatten()))(concat106_1)
relu107_1 = ReLU()(conv107_1)
resize107_1 = resize_images(relu107_1, 2, 2, 'channels_last', interpolation='bilinear')
concat107_1 = Concatenate(axis=-1)([relu102_1, resize107_1])

# Block_108
conv108_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1724_mean_Fused_Mul_2799828000_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2652726532_copy_const.npy').flatten()))(concat107_1)
relu108_1 = ReLU()(conv108_1)
resize108_1 = resize_images(relu108_1, 2, 2, 'channels_last', interpolation='bilinear')
concat108_1 = Concatenate(axis=-1)([relu101_1, resize108_1])

# Block_109
conv109_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1745_mean_Fused_Mul_2800228004_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2653526540_copy_const.npy').flatten()))(concat108_1)
relu109_1 = ReLU()(conv109_1)
resize109_1 = resize_images(relu109_1, 2, 2, 'channels_last', interpolation='bilinear')
concat109_1 = Concatenate(axis=-1)([relu100_1, resize109_1])

# Block_110
conv110_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1766_mean_Fused_Mul_2800628008_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2654326548_copy_const.npy').flatten()))(concat109_1)
relu110_1 = ReLU()(conv110_1)
resize110_1 = resize_images(relu110_1, 2, 2, 'channels_last', interpolation='bilinear')
concat110_1 = Concatenate(axis=-1)([relu99_1, resize110_1])

# Block_111
conv111_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1787_mean_Fused_Mul_2801028012_const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/data_add_2655126556_copy_const.npy').flatten()))(concat110_1)
relu111_1 = ReLU()(conv111_1)
add111_2 = Add()([relu98_1, relu111_1])
conv111_2 = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/onnx_initializer_node_side1.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1790_Dims15867_copy_const.npy').flatten()))(add111_2)
sigm111_1 = tf.math.sigmoid(conv111_2, name='sigmd1')


# Block_112
conv112_1 = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/onnx_initializer_node_side6.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1863_Dims15759_copy_const.npy').flatten()))(add60_2)
resize112_1 = resize_images(conv112_1, 32, 32, 'channels_last', interpolation='bilinear')
sigm112_1 = tf.math.sigmoid(resize112_1, name='sigmd6')

# Block_113
conv113_1 = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/onnx_initializer_node_side5.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1845_Dims15729_copy_const.npy').flatten()))(add67_2)
resize113_1 = resize_images(conv113_1, 16, 16, 'channels_last', interpolation='bilinear')
sigm113_1 = tf.math.sigmoid(resize113_1, name='sigmd5')

# Block_114
conv114_1 = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/onnx_initializer_node_side4.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1827_Dims16323_copy_const.npy').flatten()))(add75_2)
resize114_1 = resize_images(conv114_1, 8, 8, 'channels_last', interpolation='bilinear')
sigm114_1 = tf.math.sigmoid(resize114_1, name='sigmd4')

# Block_115
conv115_1 = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/onnx_initializer_node_side3.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1809_Dims15945_copy_const.npy').flatten()))(add85_2)
resize115_1 = resize_images(conv115_1, 4, 4, 'channels_last', interpolation='bilinear')
sigm115_1 = tf.math.sigmoid(resize115_1, name='sigmd3')

# Block_116
conv116_1 = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/onnx_initializer_node_side2.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1791_Dims16341_copy_const.npy').flatten()))(add97_2)
resize116_1 = resize_images(conv116_1, 2, 2, 'channels_last', interpolation='bilinear')
sigm116_1 = tf.math.sigmoid(resize116_1, name='sigmd2')

# Block_117
concat117_1 = Concatenate(axis=-1)([conv111_2, resize112_1, resize113_1, resize114_1, resize115_1, resize116_1])
conv117_1 = Conv2D(filters=1, kernel_size=[1, 1], strides=[1, 1], padding="valid", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_u2netp/480x640/FP32/onnx_initializer_node_outconv.weight_Output_0_Data__const.npy').transpose(2,3,1,0)),
                 bias_initializer=Constant(np.load('weights_u2netp/480x640/FP32/1882_Dims16215_copy_const.npy').flatten()))(concat117_1)
sigm117_1 = tf.math.sigmoid(conv117_1, name='sigmd0')



model = Model(inputs=inputs, outputs=[sigm117_1]) #, sigm111_1, sigm112_1, sigm113_1, sigm114_1, sigm115_1, sigm116_1])

model.summary()

tf.saved_model.save(model, 'saved_model_{}x{}'.format(height, width))
model.save('u2netp_{}x{}_float32.h5'.format(height, width))

full_model = tf.function(lambda inputs: model(inputs))
full_model = full_model.get_concrete_function(inputs = (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)))
frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
frozen_func.graph.as_graph_def()
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=".",
                    name="u2netp_{}x{}_float32.pb".format(height, width),
                    as_text=False)

# No Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('u2netp_{}x{}_float32.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print("tflite convert complete! - u2netp_{}x{}_float32.tflite".format(height, width))


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
with open('u2netp_{}x{}_weight_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Weight Quantization complete! - u2netp_{}x{}_weight_quant.tflite'.format(height, width))


def representative_dataset_gen():
    for data in raw_test_data.take(10):
        image = data['image'].numpy()
        image = tf.image.resize(image, (height, width))
        image = image[np.newaxis,:,:,:]
        image = image - 127.5
        image = image * 0.007843
        yield [image]

raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)


# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('u2netp_{}x{}_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_quant_model)
print('Integer Quantization complete! - u2netp_{}x{}_integer_quant.tflite'.format(height, width))


# Full Integer Quantization - Input/Output=int8
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('u2netp_{}x{}_full_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_quant_model)
print('Integer Quantization complete! - u2netp_{}x{}_full_integer_quant.tflite'.format(height, width))


# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('u2netp_{}x{}_float16_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_quant_model)
print('Float16 Quantization complete! - u2netp_{}x{}_float16_quant.tflite'.format(height, width))


# EdgeTPU
import subprocess
result = subprocess.check_output(["edgetpu_compiler", "-s", "u2netp_{}x{}_full_integer_quant.tflite".format(height, width)])
print(result)
