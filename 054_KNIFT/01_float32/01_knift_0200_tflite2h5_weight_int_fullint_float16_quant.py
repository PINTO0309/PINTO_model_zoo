### tensorflow==2.3.0

### https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/Add
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/Concatenate
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D
### https://www.tensorflow.org/api_docs/python/tf/keras/backend/l2_normalize
### https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer

### How to initialize a convolution layer with an arbitrary kernel in Keras? https://stackoverrun.com/ja/q/12269118

###  saved_model_cli show --dir saved_model_0200/ --tag_set serve --signature_def serving_default

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, ReLU, MaxPool2D, Reshape, Concatenate, AveragePooling2D, Layer
from tensorflow.keras.backend import l2_normalize
import keras
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

inputs = Input(shape=(32, 32, 1), batch_size=200, name='rgb_to_grayscale_1')

# Block_01
depthconv1_1 = DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", depth_multiplier=16, dilation_rate=[1, 1],
                 depthwise_initializer=Constant(np.load('weights_200/siamese_neural_congas_feature_extraction_Conv_weights')),
                 bias_initializer=Constant(np.load('weights_200/siamese_neural_congas_1_feature_extraction_Conv_Conv2D_bias')))(inputs)
relu1_1 = ReLU(max_value=6.)(depthconv1_1)
conv1_1 = Conv2D(filters=32, kernel_size=[3, 3], strides=[2, 2], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_200/siamese_neural_congas_feature_extraction_Conv_1_weights').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_200/siamese_neural_congas_1_feature_extraction_Conv_1_Conv2D_bias')))(relu1_1)
relu1_2 = ReLU(max_value=6.)(conv1_1)
conv1_2 = Conv2D(filters=32, kernel_size=[1, 1], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_200/siamese_neural_congas_feature_extraction_Conv_2_weights').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_200/siamese_neural_congas_1_feature_extraction_Conv_2_Conv2D_bias')))(relu1_2)
relu1_3 = ReLU(max_value=6.)(conv1_2)

# Block_02
conv2_1 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_200/siamese_neural_congas_Mixed_6a_Branch_0_Conv2d_0a_1x1_weights').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_200/siamese_neural_congas_1_Mixed_6a_Branch_0_Conv2d_0a_1x1_Conv2D_bias')))(relu1_3)
relu2_1 = ReLU(max_value=6.)(conv2_1)
conv2_2 = Conv2D(filters=16, kernel_size=[3, 3], strides=[2, 2], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_200/siamese_neural_congas_Mixed_6a_Branch_0_Conv2d_0b_3x3_weights').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_200/siamese_neural_congas_1_Mixed_6a_Branch_0_Conv2d_0b_3x3_Conv2D_bias')))(relu2_1)
relu2_2 = ReLU(max_value=6.)(conv2_2)

conv2_3 = Conv2D(filters=16, kernel_size=[1, 1], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_200/siamese_neural_congas_Mixed_6a_Branch_1_Conv2d_1a_1x1_weights').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_200/siamese_neural_congas_1_Mixed_6a_Branch_1_Conv2d_1a_1x1_Conv2D_bias')))(relu1_3)
relu2_3 = ReLU(max_value=6.)(conv2_3)
conv2_4 = Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_200/siamese_neural_congas_Mixed_6a_Branch_1_Conv2d_1b_3x3_weights').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_200/siamese_neural_congas_1_Mixed_6a_Branch_1_Conv2d_1b_3x3_Conv2D_bias')))(relu2_3)
relu2_4 = ReLU(max_value=6.)(conv2_4)
conv2_5 = Conv2D(filters=16, kernel_size=[3, 3], strides=[2, 2], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_200/siamese_neural_congas_Mixed_6a_Branch_1_Conv2d_1c_3x3_weights').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_200/siamese_neural_congas_1_Mixed_6a_Branch_1_Conv2d_1c_3x3_Conv2D_bias')))(relu2_4)
relu2_5 = ReLU(max_value=6.)(conv2_5)

maxpool2_1 = MaxPool2D(pool_size=[3, 3], strides=[2, 2], padding='same')(relu1_3)

concat2_1 = Concatenate(axis=3)([relu2_2, relu2_5, maxpool2_1])

# Block_03
conv3_1 = Conv2D(filters=32, kernel_size=[1, 1], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_200/siamese_neural_congas_Mixed7a_Branch_0_Conv2d_0a_1x1_weights').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_200/siamese_neural_congas_1_Mixed7a_Branch_0_Conv2d_0a_1x1_Conv2D_bias')))(concat2_1)
relu3_1 = ReLU(max_value=6.)(conv3_1)
conv3_2 = Conv2D(filters=32, kernel_size=[3, 3], strides=[2, 2], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_200/siamese_neural_congas_Mixed7a_Branch_0_Conv2d_0b_3x3_weights').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_200/siamese_neural_congas_1_Mixed7a_Branch_0_Conv2d_0b_3x3_Conv2D_bias')))(relu3_1)
relu3_2 = ReLU(max_value=6.)(conv3_2)

conv3_3 = Conv2D(filters=32, kernel_size=[1, 1], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_200/siamese_neural_congas_Mixed7a_Branch_1_Conv2d_1a_1x1_weights').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_200/siamese_neural_congas_1_Mixed7a_Branch_1_Conv2d_1a_1x1_Conv2D_bias')))(concat2_1)
relu3_3 = ReLU(max_value=6.)(conv3_3)
conv3_4 = Conv2D(filters=32, kernel_size=[1, 7], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_200/siamese_neural_congas_Mixed7a_Branch_1_Conv2d_1b_1x7_weights').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_200/siamese_neural_congas_1_Mixed7a_Branch_1_Conv2d_1b_1x7_Conv2D_bias')))(relu3_3)
relu3_4 = ReLU(max_value=6.)(conv3_4)
conv3_5 = Conv2D(filters=32, kernel_size=[7, 1], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_200/siamese_neural_congas_Mixed7a_Branch_1_Conv2d_1c_7x1_weights').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_200/siamese_neural_congas_1_Mixed7a_Branch_1_Conv2d_1c_7x1_Conv2D_bias')))(relu3_4)
relu3_5 = ReLU(max_value=6.)(conv3_5)
conv3_6 = Conv2D(filters=32, kernel_size=[3, 3], strides=[2, 2], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_200/siamese_neural_congas_Mixed7a_Branch_1_Conv2d_1d_3x3_weights').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_200/siamese_neural_congas_1_Mixed7a_Branch_1_Conv2d_1d_3x3_Conv2D_bias')))(relu3_5)
relu3_6 = ReLU(max_value=6.)(conv3_6)

maxpool3_1 = MaxPool2D(pool_size=[3, 3], strides=[2, 2], padding='same')(concat2_1)

concat3_1 = Concatenate(axis=3)([relu3_2, relu3_6, maxpool3_1])

# Block_04
avgpool4_1 = AveragePooling2D(pool_size=[4, 4], strides=[1, 1], padding='valid')(concat3_1)
conv4_1 = Conv2D(filters=40, kernel_size=[1, 1], strides=[1, 1], padding="same", dilation_rate=[1, 1],
                 kernel_initializer=Constant(np.load('weights_200/siamese_neural_congas_feature_compression_Conv2d_0a_weights').transpose(1,2,3,0)),
                 bias_initializer=Constant(np.load('weights_200/siamese_neural_congas_1_feature_compression_Conv2d_0a_Conv2D_bias')))(avgpool4_1)
reshape4_1 = tf.reshape(conv4_1, (200, 40))
l2norm4_1 = l2_normalize(reshape4_1)
normalize_embeddings = keras.backend.identity(l2norm4_1, name='normalize_embeddings')

model = Model(inputs=inputs, outputs=[normalize_embeddings])

model.summary()

tf.saved_model.save(model, 'saved_model_0200')
model.save('knift_0200.h5')


# No Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('knift_0200_32x32_float32.tflite', 'wb') as w:
    w.write(tflite_model)
print("tflite convert complete! - knift_0200_32x32_float32.tflite")


# Weight Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
with open('knift_0200_32x32_weight_quant.tflite', 'wb') as w:
    w.write(tflite_model)
print("Weight Quantization complete! - knift_0200_32x32_weight_quant.tflite")


def representative_dataset_gen():
    imagelist = None
    for data in raw_test_data:
        image = data['image'].numpy()
        image = tf.image.resize(image, (32, 32))
        image = tf.image.rgb_to_grayscale(image)
        image = image[np.newaxis,:,:,:]
        # image = image - 127.5
        # image = image * 0.007843
        image = image / 255.0
        if imagelist is None:
            imagelist = np.asarray(image)
        else:
            imagelist = np.vstack((imagelist, image))
        if imagelist.shape[0] == 200:
            yield [imagelist]
            imagelist = None

raw_test_data, info = tfds.load(name="cifar10", with_info=True, split="test", data_dir="~/TFDS", download=True)


# Integer Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()
with open('knift_0200_32x32_integer_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Integer Quantization complete! - knift_0200_32x32_integer_quant.tflite")


# # Full Integer Quantization - Input/Output=int8
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# converter.representative_dataset = representative_dataset_gen
# tflite_quant_model = converter.convert()
# with open('knift_0200_32x32_full_integer_quant.tflite', 'wb') as w:
#     w.write(tflite_quant_model)
# print("Integer Quantization complete! - knift_0200_32x32_full_integer_quant.tflite")


# Float16 Quantization - Input/Output=float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('knift_0200_32x32_float16_quant.tflite', 'wb') as w:
    w.write(tflite_quant_model)
print("Float16 Quantization complete! - knift_0200_32x32_float16_quant.tflite")


# # EdgeTPU
# import subprocess
# result = subprocess.check_output(["edgetpu_compiler", "-s", "knift_0200_32x32_full_integer_quant.tflite"])
# print(result)
