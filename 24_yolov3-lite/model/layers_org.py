# coding: utf-8

import tensorflow as tf


def batch_normalization(input_data, input_c, training, decay=0.9):
    """
    :param input_data: format is 'NHWC'
    :param input_c: channel of input_data
    :param training: 是否在训练，即bn会根据该参数选择mean and variance
    :param decay: 均值方差滑动参数
    :return: BN后的数据
    """
    with tf.variable_scope('BatchNorm'):
        gamma = tf.get_variable(name='gamma', shape=input_c, dtype=tf.float32,
                                initializer=tf.ones_initializer, trainable=True)
        beta = tf.get_variable(name='beta', shape=input_c, dtype=tf.float32,
                               initializer=tf.zeros_initializer, trainable=True)
        moving_mean = tf.get_variable(name='moving_mean', shape=input_c, dtype=tf.float32,
                                      initializer=tf.zeros_initializer, trainable=False)
        moving_variance = tf.get_variable(name='moving_variance', shape=input_c, dtype=tf.float32,
                                          initializer=tf.ones_initializer, trainable=False)

        def mean_and_var_update():
            axes = (0, 1, 2)
            batch_mean = tf.reduce_mean(input_data, axis=axes)
            batch_var = tf.reduce_mean(tf.pow(input_data - batch_mean, 2), axis=axes)
            with tf.control_dependencies([tf.assign(moving_mean, moving_mean * decay + batch_mean * (1 - decay)),
                                          tf.assign(moving_variance, moving_variance * decay + batch_var * (1 - decay))]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, variance = tf.cond(training, mean_and_var_update, lambda: (moving_mean, moving_variance))
        return tf.nn.batch_normalization(input_data, mean, variance, beta, gamma, 1e-05)


def group_normalization(input_data, input_c, num_group=32, eps=1e-5):
    """
    :param input_data: format is 'NHWC'，C必须是num_group的整数倍
    :param input_c: channel of input_data
    :return: GN后的数据
    """
    with tf.variable_scope('GroupNorm'):
        input_shape = tf.shape(input_data)
        N = input_shape[0]
        H = input_shape[1]
        W = input_shape[2]
        C = input_c
        assert (C % num_group) == 0
        input_data = tf.reshape(input_data, (N, H, W, num_group, C // num_group))
        axes = (1, 2, 4)
        mean = tf.reduce_mean(input_data, axis=axes, keep_dims=True)
        std = tf.sqrt(tf.reduce_mean(tf.pow(input_data - mean, 2), axis=axes, keep_dims=True) + eps)
        input_data = 1.0 * (input_data - mean) / std
        input_data = tf.reshape(input_data, (N, H, W, C))
        gamma = tf.get_variable(name='gamma', shape=C, dtype=tf.float32,
                                initializer=tf.ones_initializer, trainable=True)
        beta = tf.get_variable(name='beta', shape=C, dtype=tf.float32,
                               initializer=tf.zeros_initializer, trainable=True)
    return gamma * input_data + beta


def convolutional(name, input_data, filters_shape, training, downsample=False, activate=True, bn=True):
    """
    :param name: convolutional layer 的名字
    :param input_data: shape为(batch, height, width, channels)
    :param filters_shape: shape为(filter_height, filter_width, filter_channel, filter_num)
    :param training: 必须是tensor，True or False
    :param downsample: 是否对输入进行下采样
    :param activate: 是否使用激活函数
    :param bn: 是否使用batch normalization
    :return:
    """
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

        weight = tf.get_variable(name='weights', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)
        if bn:
            conv = batch_normalization(input_data=conv, input_c=filters_shape[-1], training=training)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)
        if activate:
            conv = tf.nn.relu6(conv)
    return conv

def separable_conv(name, input_data, input_c, output_c, training, downsample=False):
    """
    :param name:
    :param input_data: shape 为NHWC
    :param input_c: channel of input data
    :param output_c: channel of output data
    :param training: 是否在训练，必须为tensor
    :param downsample: 是否下采样
    :return: 输出数据的shape为(N, H, W, output_channel)
    """
    with tf.variable_scope(name):
        with tf.variable_scope('depthwise'):
            if downsample:
                pad_data = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
                input_data = tf.pad(input_data, pad_data, 'CONSTANT')
                strides = (1, 2, 2, 1)
                padding = 'VALID'
            else:
                strides = (1, 1, 1, 1)
                padding = "SAME"
            dwise_weight = tf.get_variable(name='depthwise_weights', dtype=tf.float32, trainable=True,
                                           shape=(3, 3, input_c, 1),
                                           initializer=tf.random_normal_initializer(stddev=0.01))
            dwise_conv = tf.nn.depthwise_conv2d(input=input_data, filter=dwise_weight, strides=strides, padding=padding)
            dwise_conv = batch_normalization(input_data=dwise_conv, input_c=input_c, training=training)
            dwise_conv = tf.nn.relu6(dwise_conv)

        with tf.variable_scope('pointwise'):
            pwise_weight = tf.get_variable(name='pointwise_weights', dtype=tf.float32, trainable=True,
                                           shape=(1, 1, input_c, output_c),
                                           initializer=tf.random_normal_initializer(stddev=0.01))
            pwise_conv = tf.nn.conv2d(input=dwise_conv, filter=pwise_weight, strides=(1, 1, 1, 1), padding="SAME")
            pwise_conv = batch_normalization(input_data=pwise_conv, input_c=output_c, training=training)
            pwise_conv = tf.nn.relu6(pwise_conv)
        return pwise_conv


def inverted_residual(name, input_data, input_c, output_c, training, downsample=False, t=6):
    """
    :param name:
    :param input_data: shape 为NHWC
    :param input_c: channel of input data
    :param output_c: channel of output data
    :param training: 是否在训练，必须为tensor
    :param downsample: 是否下采样
    :param t: expansion factor
    :return: 输出数据的shape为(N, H, W, output_channel)
    """
    with tf.variable_scope(name):
        expand_c = t * input_c

        with tf.variable_scope('expand'):
            if t > 1:
                expand_weight = tf.get_variable(name='weights', dtype=tf.float32, trainable=True,
                                                shape=(1, 1, input_c, expand_c),
                                                initializer=tf.random_normal_initializer(stddev=0.01))
                expand_conv = tf.nn.conv2d(input=input_data, filter=expand_weight, strides=(1, 1, 1, 1), padding="SAME")
                expand_conv = batch_normalization(input_data=expand_conv, input_c=expand_c, training=training)
                expand_conv = tf.nn.relu6(expand_conv)
            else:
                expand_conv = input_data

        with tf.variable_scope('depthwise'):
            if downsample:
                pad_data = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
                expand_conv = tf.pad(expand_conv, pad_data, 'CONSTANT')
                strides = (1, 2, 2, 1)
                padding = 'VALID'
            else:
                strides = (1, 1, 1, 1)
                padding = "SAME"
            dwise_weight = tf.get_variable(name='depthwise_weights', dtype=tf.float32, trainable=True,
                                           shape=(3, 3, expand_c, 1),
                                           initializer=tf.random_normal_initializer(stddev=0.01))
            dwise_conv = tf.nn.depthwise_conv2d(input=expand_conv, filter=dwise_weight, strides=strides, padding=padding)
            dwise_conv = batch_normalization(input_data=dwise_conv, input_c=expand_c, training=training)
            dwise_conv = tf.nn.relu6(dwise_conv)

        with tf.variable_scope('project'):
            pwise_weight = tf.get_variable(name='weights', dtype=tf.float32, trainable=True,
                                           shape=(1, 1, expand_c, output_c),
                                           initializer=tf.random_normal_initializer(stddev=0.01))
            pwise_conv = tf.nn.conv2d(input=dwise_conv, filter=pwise_weight, strides=(1, 1, 1, 1), padding="SAME")
            pwise_conv = batch_normalization(input_data=pwise_conv, input_c=output_c, training=training)
        if downsample or pwise_conv.get_shape().as_list()[3] != input_data.get_shape().as_list()[3]:
            return pwise_conv
        else:
            return input_data + pwise_conv


def residual_block(name, input_data, input_channel, filter_num1, filter_num2, training):
    """
    :param name: residual_block的名字
    :param input_data: shape为(batch, height, width, channels)
    :param input_channel: input_data的channel
    :param filter_num1: residual block中第一个卷积层的卷积核个数
    :param filter_num2: residual block中第二个卷积层的卷积核个数
    :param training: 必须是tensor，True or False
    :return: residual block 的输出
    """
    with tf.variable_scope(name):
        conv = convolutional(name='conv1', input_data=input_data, filters_shape=(1, 1, input_channel, filter_num1),
                             training=training)
        conv = convolutional(name='conv2', input_data=conv, filters_shape=(3, 3, filter_num1, filter_num2),
                             training=training)
        residual_output = input_data + conv
    return residual_output


def pool(name, input_data, ksize=(1, 2, 2, 1), stride=(1, 2, 2, 1),
         padding='SAME', pooling=tf.nn.max_pool):
    """
    :param name: pooling层的命名空间
    :param input_data: pooling层的输入数据，格式为'NHWC'
    :param ksize: pooling的size，格式为[1, pooling_height, pooling_width, 1]
    :param stride: pooling的stride，格式为[1, stride, stride, 1]
    :param padding: 是否padding 'SAME' or 'VALID'
    :param pooling: 选择用哪个pooling层
    :return: 池化后的数据
    """
    with tf.variable_scope(name):
        pool_output = pooling(value=input_data, ksize=ksize, strides=stride, padding=padding)
    return pool_output


def route(name, previous_output, current_output):
    """
    :param name: route层的名字
    :param previous_output: 前面层的输出
    :param current_output: 当前层的输出
    :return:
    """
    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)
    return output


def upsample(name, input_data):
    """
    :param name: upsample层的名字
    :param input_data: shape为(batch, height, width, channels)
    :return:
    """
    with tf.variable_scope(name):
        input_shape = tf.shape(input_data)
        output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))
    return output

def decode(name, conv_output, num_classes, stride):
    """
    :param conv_output: yolo的输出，shape为(batch_size, output_size, output_size, gt_per_grid * (5 + num_classes))
    :param num_classes: 类别的数量
    :param stride: YOLO的stride
    :return:
    pred_bbox: shape为(batch_size, output_size, output_size, gt_per_grid, 5 + num_classes)
    5 + num_classes指的是预测bbox的(xmin, ymin, xmax, ymax, confidence, probability)
    其中(xmin, ymin, xmax, ymax)是预测bbox的左上角和右下角坐标，大小是相对于input_size的，
    confidence是预测bbox属于物体的概率，probability是条件概率分布
    """
    with tf.variable_scope(name):
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        gt_per_grid = conv_shape[3] / (5 + num_classes)

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, gt_per_grid, 5 + num_classes))
        conv_raw_dx1dy1 = conv_output[:, :, :, :, 0:2]
        conv_raw_dx2dy2 = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        # 获取yolo的输出feature map中每个grid左上角的坐标
        # 需注意的是图像的坐标轴方向为
        #  - - - - > x
        # |
        # |
        # ↓
        # y
        # 在图像中标注坐标时通常用(y,x)，但此处为了与coor的存储格式(dx, dy, dw, dh)保持一致，将grid的坐标存储为(x, y)的形式

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, gt_per_grid, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        # (1)对xmin, ymin, xmax, ymax进行decode
        # dx_min, dy_min = exp(rawdx1dy1)
        # dx_max, dy_max = exp(rawdx2dy2)
        # xmin, ymin = ((x_grid, y_grid) + 0.5 - (dx_min, dy_min)) * stride
        # xmax, ymax = ((x_grid, y_grid) + 0.5 + (dx_max, dy_max)) * stride
        pred_xymin = (xy_grid + 0.5 - tf.exp(conv_raw_dx1dy1)) * stride
        pred_xymax = (xy_grid + 0.5 + tf.exp(conv_raw_dx2dy2)) * stride
        pred_corner = tf.concat([pred_xymin, pred_xymax], axis=-1)

        # (2)对confidence进行decode
        pred_conf = tf.sigmoid(conv_raw_conf)

        # (3)对probability进行decode
        pred_prob = tf.sigmoid(conv_raw_prob)

        pred_bbox = tf.concat([pred_corner, pred_conf, pred_prob], axis=-1)
        return pred_bbox

