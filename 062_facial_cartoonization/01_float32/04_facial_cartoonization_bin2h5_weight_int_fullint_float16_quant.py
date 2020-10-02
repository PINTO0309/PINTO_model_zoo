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
### https://www.tensorflow.org/api_docs/python/tf/keras/backend/resize_images
### https://www.tensorflow.org/api_docs/python/tf/keras/activations/tanh

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
from tensorflow.keras.activations import tanh
import numpy as np
import sys
import tensorflow_datasets as tfds

# tmp = np.load('weights/256x256/FP32/depthwise_conv2d_Kernel')
# print(tmp.shape)
# print(tmp)

# def init_f(shape, dtype=None):
#        ker = np.load('weights/256x256/FP32/depthwise_conv2d_Kernel')
#        print(shape)
#        return ker

# sys.exit(0)


height = 256
width  = 256
inputs = Input(shape=(height, width, 3), batch_size=1, name='input')

# Block_01
conv1_1 = Conv2D(filters=32, kernel_size=[3, 3], strides=[2, 2], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/128_mean_Fused_Mul_49864988_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_4610_copy_const.npy').transpose(0,2,3,1).flatten()))(inputs)
conv1_2 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/131_mean_Fused_Mul_49904992_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_46134618_copy_const.npy').transpose(0,2,3,1).flatten()))(conv1_1)

# Block_02
conv2_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[2, 2], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/134_mean_Fused_Mul_49944996_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_46214626_copy_const.npy').transpose(0,2,3,1).flatten()))(conv1_2)
conv2_2 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/137_mean_Fused_Mul_49985000_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_46294634_copy_const.npy').transpose(0,2,3,1).flatten()))(conv2_1)

# Block_03
conv3_1 = Conv2D(filters=96, kernel_size=[3, 3], strides=[2, 2], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/140_mean_Fused_Mul_50025004_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_46374642_copy_const.npy').transpose(0,2,3,1).flatten()))(conv2_2)
conv3_2 = Conv2D(filters=96, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/143_mean_Fused_Mul_50065008_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_46454650_copy_const.npy').transpose(0,2,3,1).flatten()))(conv3_1)

# Block_04
conv4_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[2, 2], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/146_mean_Fused_Mul_50105012_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_46534658_copy_const.npy').transpose(0,2,3,1).flatten()))(conv3_2)
conv4_2 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/149_mean_Fused_Mul_50145016_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_46614666_copy_const.npy').transpose(0,2,3,1).flatten()))(conv4_1)

# Block_05
conv5_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/152_mean_Fused_Mul_50185020_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_46694674_copy_const.npy').transpose(0,2,3,1).flatten()))(conv4_2)
conv5_2 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/155_mean_Fused_Mul_50225024_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_46774682_copy_const.npy').transpose(0,2,3,1).flatten()))(conv5_1)
add5_1 = Add()([conv4_2, conv5_2])
relu5_1 = ReLU()(add5_1)

# Block_06
conv6_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/159_mean_Fused_Mul_50265028_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_46854690_copy_const.npy').transpose(0,2,3,1).flatten()))(relu5_1)
conv6_2 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/162_mean_Fused_Mul_50305032_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_46934698_copy_const.npy').transpose(0,2,3,1).flatten()))(conv6_1)
add6_1 = Add()([relu5_1, conv6_2])
relu6_1 = ReLU()(add6_1)

# Block_07
conv7_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/166_mean_Fused_Mul_50345036_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_47014706_copy_const.npy').transpose(0,2,3,1).flatten()))(relu6_1)
conv7_2 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/169_mean_Fused_Mul_50385040_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_47094714_copy_const.npy').transpose(0,2,3,1).flatten()))(conv7_1)
add7_1 = Add()([relu6_1, conv7_2])
relu7_1 = ReLU()(add7_1)

# Block_08
conv8_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/173_mean_Fused_Mul_50425044_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_47174722_copy_const.npy').transpose(0,2,3,1).flatten()))(relu7_1)
conv8_2 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/176_mean_Fused_Mul_50465048_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_47254730_copy_const.npy').transpose(0,2,3,1).flatten()))(conv8_1)
add8_1 = Add()([relu7_1, conv8_2])
relu8_1 = ReLU()(add8_1)

# Block_09
conv9_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/180_mean_Fused_Mul_50505052_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_47334738_copy_const.npy').transpose(0,2,3,1).flatten()))(relu8_1)
resize9_1 = resize_images(conv9_1, 2, 2, 'channels_last', interpolation='nearest')
conv9_2 = Conv2D(filters=96, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/192_mean_Fused_Mul_50545056_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_47414746_copy_const.npy').transpose(0,2,3,1).flatten()))(resize9_1)
add9_1 = Add()([conv3_2, conv9_2])

# Block_10
conv10_1 = Conv2D(filters=96, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/196_mean_Fused_Mul_50585060_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_47494754_copy_const.npy').transpose(0,2,3,1).flatten()))(add9_1)
resize10_1 = resize_images(conv10_1, 2, 2, 'channels_last', interpolation='nearest')
conv10_2 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/208_mean_Fused_Mul_50625064_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_47574762_copy_const.npy').transpose(0,2,3,1).flatten()))(resize10_1)
add10_1 = Add()([conv2_2, conv10_2])

# Block_11
conv11_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/212_mean_Fused_Mul_50665068_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_47654770_copy_const.npy').transpose(0,2,3,1).flatten()))(add10_1)
resize11_1 = resize_images(conv11_1, 2, 2, 'channels_last', interpolation='nearest')
conv11_2 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/224_mean_Fused_Mul_50705072_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_47734778_copy_const.npy').transpose(0,2,3,1).flatten()))(resize11_1)
add11_1 = Add()([conv1_2, conv11_2])

# Block_12
conv12_1 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1], activation='relu',
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/228_mean_Fused_Mul_50745076_const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/data_add_47814786_copy_const.npy').transpose(0,2,3,1).flatten()))(add11_1)
resize12_1 = resize_images(conv12_1, 2, 2, 'channels_last', interpolation='nearest')
conv12_2 = Conv2D(filters=3, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/256x256/FP32/onnx_initializer_node_up4.conv_layer.4.weight_Output_0_Data__const.npy').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights/256x256/FP32/239_Dims2647_copy_const.npy').transpose(0,2,3,1).flatten()))(resize12_1)
tanh12_1 = tanh(conv12_2)


model = Model(inputs=inputs, outputs=[tanh12_1])

model.summary()

tf.saved_model.save(model, 'saved_model_{}x{}'.format(height, width))
model.save('facial_cartoonization_{}x{}_float32.h5'.format(height, width))


# No Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('facial_cartoonization_{}x{}_float32.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print("tflite convert complete! - facial_cartoonization_{}x{}_float32.tflite".format(height, width))


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
with open('facial_cartoonization_{}x{}_weight_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_model)
print('Weight Quantization complete! - facial_cartoonization_{}x{}_weight_quant.tflite'.format(height, width))


def representative_dataset_gen():
    for data in raw_test_data.take(100):
        image = data['image'].numpy()
        image = tf.image.resize(image, (height, width))
        image = image[np.newaxis,:,:,:]
        image = image / 127.5 - 1.0
        yield [image]

raw_test_data, info = tfds.load(name="coco/2017", with_info=True, split="test", data_dir="~/TFDS", download=False)


# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('facial_cartoonization_{}x{}_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_quant_model)
print('Integer Quantization complete! - facial_cartoonization_{}x{}_integer_quant.tflite'.format(height, width))


# Full Integer Quantization - Input/Output=int8
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('facial_cartoonization_{}x{}_full_integer_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_quant_model)
print('Integer Quantization complete! - facial_cartoonization_{}x{}_full_integer_quant.tflite'.format(height, width))


# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('facial_cartoonization_{}x{}_float16_quant.tflite'.format(height, width), 'wb') as w:
    w.write(tflite_quant_model)
print('Float16 Quantization complete! - facial_cartoonization_{}x{}_float16_quant.tflite'.format(height, width))


# EdgeTPU
import subprocess
result = subprocess.check_output(["edgetpu_compiler", "-s", "facial_cartoonization_{}x{}_full_integer_quant.tflite".format(height, width)])
print(result)
