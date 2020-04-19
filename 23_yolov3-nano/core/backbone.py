#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backbone.py
#   Author      : YunYang1994
#   Created date: 2019-02-17 11:03:35
#   Description :
#
#================================================================

import core.common as common
import tensorflow as tf
import core.dataset as dataset

def darknet53_whole(input_data, trainable):

    with tf.variable_scope('darknet'):

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_data = common.residual_block(input_data,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, name='conv4', downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 128,  64, 128, trainable=trainable, name='residual%d' %(i+1))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv9', downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' %(i+3))

        route_1 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                          trainable=trainable, name='conv26', downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' %(i+11))

        route_2 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                          trainable=trainable, name='conv43', downsample=True)

        for i in range(4):
            input_data = common.residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+19))

        return route_1, route_2, input_data

def darknet53(input_data, trainable):

    with tf.variable_scope('darknet'):

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  3,  12), trainable=trainable, name='conv0')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 12,  24),
                                          trainable=trainable, name='conv1', downsample=True)
        #print(input_data.shape)
        input_data=common.PEP(input_data,24,24,7,name='PEP0',stride=1)
        input_data=common.EP(input_data,24,70,name='EP0',stride=2)
        input_data=common.PEP(input_data,70,70,25,name='PEP1',stride=1)
        input_data=common.PEP(input_data,70,70,24,name='PEP2',stride=1)
        input_data=common.EP(input_data,70,150,name='EP1',stride=2)
        input_data=common.PEP(input_data,150,150,56,name='PEP3',stride=1)
		
        input_data = common.convolutional(input_data, filters_shape=(1, 1,  150,  150), trainable=trainable, name='conv2')
        input_data=common.FCA(input_data,150,8)
        input_data=common.PEP(input_data,150,150,73,name='PEP4',stride=1)
        input_data=common.PEP(input_data,150,150,71,name='PEP5',stride=1)
        input_data=common.PEP(input_data,150,150,75,name='PEP6',stride=1)
        route_1=input_data
		
        input_data=common.EP(input_data,150,325,name='EP2',stride=2)
        input_data=common.PEP(input_data,325,325,132,name='PEP7',stride=1)
        input_data=common.PEP(input_data,325,325,124,name='PEP8',stride=1)
        input_data=common.PEP(input_data,325,325,141,name='PEP9',stride=1)
        input_data=common.PEP(input_data,325,325,140,name='PEP10',stride=1)
        input_data=common.PEP(input_data,325,325,137,name='PEP11',stride=1)
        input_data=common.PEP(input_data,325,325,135,name='PEP12',stride=1)
        input_data=common.PEP(input_data,325,325,133,name='PEP13',stride=1)
        input_data=common.PEP(input_data,325,325,140,name='PEP14',stride=1)
        route_2=input_data
		
        input_data=common.EP(input_data,325,545,name='EP3',stride=2)
        input_data=common.PEP(input_data,545,545,276,name='PEP15',stride=1)
        input_data = common.convolutional(input_data, filters_shape=(1, 1,  545,  230), trainable=trainable, name='conv3')
        input_data=common.EP(input_data,230,489,name='EP4',stride=1)
        input_data=common.PEP(input_data,489,469,213,name='PEP16',stride=1)
        input_data = common.convolutional(input_data, filters_shape=(1, 1,  469,  189), trainable=trainable, name='conv4')
		
        return route_1, route_2, input_data


