#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : common.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 09:56:29
#   Description :
#
#================================================================

import math
import tensorflow as tf
slim = tf.contrib.slim

def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):

    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                   trainable=trainable, name='conv2')

        residual_output = input_data + short_cut

    return residual_output



def route(name, previous_output, current_output):

    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output


def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output


############yolo_nano################
def DW_convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        #conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)
        conv = tf.nn.depthwise_conv2d(input_data,filter=weight,strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv

@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      mode: The mode for tf.pad.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2#“//”表示整数出发，返回一个不大于结果的整数值
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs

def _avgpool_fixed_padding(inputs, kernel_size, strides=1):
    if strides!=1: 
        inputs = _fixed_padding(inputs, kernel_size)
        padding= 'VALID'
    else:
        padding= 'SAME'	
    #print(type(kernel_size))
    kernel_sizes=(kernel_size,kernel_size)
    #inputs = slim.avg_pool2d(inputs, kernel_sizes, stride=strides, padding=padding)
    inputs=tf.layers.average_pooling2d(inputs,pool_size=kernel_sizes,strides=strides,padding=padding)
    return inputs

def AdaptiveAvgPool2d(input_data,output_size):
    h=input_data.get_shape().as_list()[1]
    if h==None:
        h=1
    #print(h)
    #h=52
    stride=h//output_size
    kernels=h-(output_size-1)*stride
    #print(kernels)
    #print(111)
    input_data=_avgpool_fixed_padding(input_data, kernels, strides=stride)
    return input_data

def sepconv(input_data,input_channels,output_channels,name,stride=1,expand_ratio=1,trainable=True):
    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channels, input_channels*expand_ratio), trainable=trainable, name='conv1')
        #print(input_data.shape)
        if stride==2:
            ifdownsample=True
        else:
            ifdownsample=False
        input_data = DW_convolutional(input_data, filters_shape=(3, 3, input_channels, expand_ratio), trainable=True, name='DWconv2', downsample=ifdownsample, activate=True, bn=True)
        #print(input_data.shape)
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channels*expand_ratio,output_channels), trainable=trainable, name='conv3',activate=True)
    return input_data

def EP(input_data,input_channels,output_channels,name,stride=1,trainable=True):
    with tf.variable_scope(name):
        x=input_data
        input_data=sepconv(input_data,input_channels,output_channels,name='sepconv1',stride=stride,expand_ratio=1,trainable=trainable)
        if stride==1 and input_channels==output_channels:
           return x+input_data
    return input_data
	
def PEP(input_data,input_channels,output_channels,middle_channels,name,stride=1,trainable=True):
    with tf.variable_scope(name):
        x=input_data
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channels,middle_channels), trainable=trainable, name='conv1',activate=True)
        #print(input_data.shape)
        input_data = sepconv(input_data,middle_channels,output_channels,name='sepconv1',stride=stride,expand_ratio=1,trainable=trainable)
    if stride==1 and input_channels==output_channels:
        return x+input_data
    return input_data

def fcn_layer(input_data,input_dims,output_dims,activation=None):
    W=tf.Variable(tf.truncated_normal([input_dims,output_dims],stddev=0.1))
    input_data=tf.matmul(input_data,W);
    if activation==None:
        return input_data
    else:
        input_data=activation(input_data)
    return input_data

def FCA_A(input_data,channels,reduction_ratio):
    x=input_data
    b,h,w,c=input_data.shape
    #print(type(b))
    if b==None:
        b=1
        h=0
        w=0
    hidden_channels=channels//reduction_ratio
    input_data=AdaptiveAvgPool2d(input_data,1)
    input_data=tf.reshape(input_data,[b,c])
    input_data=fcn_layer(input_data,channels,hidden_channels,activation=tf.nn.relu)
    input_data=fcn_layer(input_data,hidden_channels,channels,activation=tf.sigmoid)
    input_data=tf.reshape(input_data,[b,1,1,c])
    #print(input_data.shape)
    input_data=tf.tile(input_data,[1,h,w,1])
    return x*input_data

def FCA(input_data,channels,reduction_ratio):
    x=input_data
    hidden_channels=channels//reduction_ratio
    input_data=tf.layers.dense(input_data,hidden_channels,activation=tf.nn.relu)
    input_data=tf.layers.dense(input_data,channels,activation=tf.nn.sigmoid)
    return x*input_data