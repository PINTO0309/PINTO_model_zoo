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
### https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/examples/digit_classifier/ml/mnist_tflite.ipynb#scrollTo=2fXStjR4mzkR

### How to initialize a convolution layer with an arbitrary kernel in Keras? https://stackoverrun.com/ja/q/12269118

###  saved_model_cli show --dir saved_model_pose_detection/ --tag_set serve --signature_def serving_default

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, ReLU, MaxPool2D, Reshape, Concatenate, Layer
from tensorflow.keras.initializers import Constant
import numpy as np
import sys
import tensorflow_datasets as tfds

# tmp = np.load('weights/depthwise_conv2d_Kernel')
# print(tmp.shape)
# print(tmp)

# def init_f(shape, dtype=None):
#        ker = np.load('weights/depthwise_conv2d_Kernel')
#        print(shape)
#        return ker

# sys.exit(0)

inputs = Input(shape=(96, 2000, 1), batch_size=1, name='input')

# Block_01
matmul1_1 = tf.math.multiply(inputs, np.load('weights/data_mul_28262830'))
add1_1 = Add()([matmul1_1, np.load('weights/data_add_28272832')])
conv1_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/95_mean_Fused_Mul_30603062_const').transpose(1,2,3,0)))(add1_1)
add1_2 = Add()([conv1_1, np.load('weights/data_add_28352840').transpose(0,2,3,1)])
relu1_1 = ReLU()(add1_2)

# Block_02
conv2_1 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/98_mean_Fused_Mul_30643066_const').transpose(1,2,3,0)))(relu1_1)
add2_1 = Add()([conv2_1, np.load('weights/data_add_28432848').transpose(0,2,3,1)])
relu2_1 = ReLU()(add2_1)

# Block_03
maxpool3_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu2_1)
conv3_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/102_mean_Fused_Mul_30683070_const').transpose(1,2,3,0)))(maxpool3_1)
add3_1 = Add()([conv3_1, np.load('weights/data_add_28512856').transpose(0,2,3,1)])
relu3_1 = ReLU()(add3_1)

# Block_04
conv4_1 = Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/105_mean_Fused_Mul_30723074_const').transpose(1,2,3,0)))(relu3_1)
add4_1 = Add()([conv4_1, np.load('weights/data_add_28592864').transpose(0,2,3,1)])
relu4_1 = ReLU()(add4_1)

# Block_05
maxpool5_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu4_1)
conv5_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/109_mean_Fused_Mul_30763078_const').transpose(1,2,3,0)))(maxpool5_1)
add5_1 = Add()([conv5_1, np.load('weights/data_add_28672872').transpose(0,2,3,1)])
relu5_1 = ReLU()(add5_1)

# Block_06
conv6_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/112_mean_Fused_Mul_30803082_const').transpose(1,2,3,0)))(relu5_1)
add6_1 = Add()([conv6_1, np.load('weights/data_add_28752880').transpose(0,2,3,1)])
relu6_1 = ReLU()(add6_1)

# Block_07
conv7_1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/115_mean_Fused_Mul_30843086_const').transpose(1,2,3,0)))(relu6_1)
add7_1 = Add()([conv7_1, np.load('weights/data_add_28832888').transpose(0,2,3,1)])
relu7_1 = ReLU()(add7_1)

# Block_08
maxpool8_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu7_1)
conv8_1 = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/121_mean_Fused_Mul_30883090_const').transpose(1,2,3,0)))(maxpool8_1)
add8_1 = Add()([conv8_1, np.load('weights/data_add_28912896').transpose(0,2,3,1)])
relu8_1 = ReLU()(add8_1)

# Block_09
conv9_1 = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/124_mean_Fused_Mul_30923094_const').transpose(1,2,3,0)))(relu8_1)
add9_1 = Add()([conv9_1, np.load('weights/data_add_28992904').transpose(0,2,3,1)])
relu9_1 = ReLU()(add9_1)

# Block_10
conv10_1 = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/127_mean_Fused_Mul_30963098_const').transpose(1,2,3,0)))(relu9_1)
add10_1 = Add()([conv10_1, np.load('weights/data_add_29072912').transpose(0,2,3,1)])
relu10_1 = ReLU()(add10_1)

# Block_11
maxpool11_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')(relu10_1)
conv11_1 = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/133_mean_Fused_Mul_31003102_const').transpose(1,2,3,0)))(maxpool11_1)
add11_1 = Add()([conv11_1, np.load('weights/data_add_29152920').transpose(0,2,3,1)])
relu11_1 = ReLU()(add11_1)

# Block_12
conv12_1 = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/136_mean_Fused_Mul_31043106_const').transpose(1,2,3,0)))(relu11_1)
add12_1 = Add()([conv12_1, np.load('weights/data_add_29232928').transpose(0,2,3,1)])
relu12_1 = ReLU()(add12_1)

# Block_13
conv13_1 = Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights/139_mean_Fused_Mul_31083110_const').transpose(1,2,3,0)))(relu12_1)
add13_1 = Add()([conv13_1, np.load('weights/data_add_29312936').transpose(0,2,3,1)])
relu13_1 = ReLU()(add13_1)

# Block_14
maxpool14_1 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(relu13_1)
reshape14_1 = tf.reshape(maxpool14_1, (1, 186, 512))
matmul14_1 = tf.linalg.matmul(reshape14_1, np.load('weights/148_1_port_transpose3707_const'), transpose_a=False, transpose_b=True)
add14_1 = Add()([matmul14_1, np.load('weights/onnx_initializer_node_93_Output_0_Data_')])
transpose14_1 = tf.transpose(add14_1, perm=[1, 0, 2], name='output')

model = Model(inputs=inputs, outputs=[transpose14_1])

model.summary()

tf.saved_model.save(model, 'saved_model')
model.save_weights('handwritten_japanese_recognition.h5')
open('handwritten_japanese_recognition.json', 'w').write(model.to_json())



# No Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('handwritten_japanese_recognition_90x2000_float32.tflite', 'wb') as w:
    w.write(tflite_model)
print("tflite convert complete! - handwritten_japanese_recognition_90x2000_float32.tflite")


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
with open('handwritten_japanese_recognition_90x2000_weight_quant.tflite', 'wb') as w:
    w.write(tflite_model)
print("Weight Quantization complete! - handwritten_japanese_recognition_90x2000_weight_quant.tflite")


# def representative_dataset_gen():
#     for data in raw_test_data.take(10):
#         image = data['image'].numpy()
#         image = tf.image.resize(image, (90, 2000))
#         image = image[np.newaxis,:,:,:]
#         image = 255 - image
#         image = image / 255.0
#         yield [image]

# raw_test_data, info = tfds.load(name="mnist", with_info=True, split="test", data_dir="~/TFDS", download=False)


# # Integer Quantization - Input/Output=float32
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('handwritten_japanese_recognition_90x2000_integer_quant.tflite', 'wb') as w:
#     w.write(tflite_quant_model)
# print("Integer Quantization complete! - handwritten_japanese_recognition_90x2000_integer_quant.tflite")


# # Full Integer Quantization - Input/Output=int8
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('handwritten_japanese_recognition_90x2000_full_integer_quant.tflite', 'wb') as w:
#     w.write(tflite_quant_model)
# print("Integer Quantization complete! - handwritten_japanese_recognition_90x2000_full_integer_quant.tflite")


# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('handwritten_japanese_recognition_90x2000_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - handwritten_japanese_recognition_90x2000_float16_quant.tflite")


# # EdgeTPU
# import subprocess
# result = subprocess.check_output(["edgetpu_compiler", "-s", "handwritten_japanese_recognition_90x2000_full_integer_quant.tflite"])
# print(result)
