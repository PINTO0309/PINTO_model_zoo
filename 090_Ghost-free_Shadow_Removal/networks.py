import tensorflow.compat.v1 as tf
# import tensorflow.contrib.slim as slim
import tf_slim as slim
import numpy as np
import os,time,cv2,scipy.io


def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

def lrelu(x):
    return tf.maximum(x*0.2,x)

def relu(x):
    return tf.maximum(0.0,x)

def nm(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x, trainable=False)

def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias


def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer


def se_block(input_feature, name, ratio=8):
    
    kernel_initializer = tf.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        # Global average pooling
        squeeze = tf.reduce_mean(input_feature, axis=[1,2], keepdims=True)   
        assert squeeze.get_shape()[1:] == (1,1,channel)
        excitation = tf.layers.dense(inputs=squeeze,
                                 units=channel//ratio,
                                 activation=tf.nn.relu,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='bottleneck_fc', trainable=False)
        assert excitation.get_shape()[1:] == (1,1,channel//ratio)
        excitation = tf.layers.dense(inputs=excitation,
                                 units=channel,
                                 activation=tf.nn.sigmoid,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='recover_fc', trainable=False)
        assert excitation.get_shape()[1:] == (1,1,channel)
        scale = input_feature * excitation    
    return scale

def build_vgg19(input,vgg_path,reuse=False):
    vgg_path=scipy.io.loadmat(vgg_path)
    with tf.variable_scope("vgg19"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        net={}
        vgg_layers=vgg_path['layers'][0]
        net['input']=input-np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
        net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
        net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
        net['pool1']=build_net('pool',net['conv1_2'])
        net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
        net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
        net['pool2']=build_net('pool',net['conv2_2'])
        net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
        net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
        net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
        net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
        net['pool3']=build_net('pool',net['conv3_4'])
        net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
        net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
        net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
        net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
        net['pool4']=build_net('pool',net['conv4_4'])
        net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28), name='vgg_conv5_1')
        net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30), name='vgg_conv5_2')
        net['conv5_3'] = build_net('conv', net['conv5_2'], get_weight_bias(vgg_layers, 32), name='vgg_conv5_3')
        net['conv5_4'] = build_net('conv', net['conv5_3'], get_weight_bias(vgg_layers, 34), name='vgg_conv5_4')
        net['pool5'] = build_net('pool', net['conv5_4'])
        return net


def spp(net,channel=64,scope='g_pool'):

    # here we build the pooling stack
    net_2 = tf.layers.average_pooling2d(net,pool_size=4,strides=4,padding='same')
    net_2 = slim.conv2d(net_2,channel,[1,1],activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope=scope+'2', trainable=False)

    net_8 = tf.layers.average_pooling2d(net,pool_size=8,strides=8,padding='same')
    net_8 = slim.conv2d(net_8,channel,[1,1],activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope=scope+'8', trainable=False)

    net_16 = tf.layers.average_pooling2d(net,pool_size=16,strides=16,padding='same')
    net_16 = slim.conv2d(net_16,channel,[1,1],activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope=scope+'16', trainable=False)

    net_32 = tf.layers.average_pooling2d(net,pool_size=32,strides=32,padding='same')
    net_32 = slim.conv2d(net_32,channel,[1,1],activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope=scope+'32', trainable=False)

    net = tf.concat([
      tf.image.resize_bilinear(net_2,(tf.shape(net)[1],tf.shape(net)[2])),
      tf.image.resize_bilinear(net_8,(tf.shape(net)[1],tf.shape(net)[2])),
      tf.image.resize_bilinear(net_16,(tf.shape(net)[1],tf.shape(net)[2])),
      tf.image.resize_bilinear(net_32,(tf.shape(net)[1],tf.shape(net)[2])),
      net],axis=3)

    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope=scope+'sf', trainable=False)

    return net

def agg(net,channel=64,scope='g_agg'):
    net = se_block(net,scope+'att2')
    net = slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope=scope+'agg2', trainable=False)
    return net


def build_aggasatt_joint(input,channel=64,vgg_19_path='None'):
    print("[i] Hypercolumn ON, building hypercolumn features ... ")
    vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0,vgg_19_path)
    for layer_id in range(1,6):
        vgg19_f = vgg19_features['conv%d_2'%layer_id]
        input = tf.concat([tf.image.resize_bilinear(vgg19_f,(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)

    sf=slim.conv2d(input,channel,[1,1],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_sf', trainable=False)

    net0=slim.conv2d(sf,channel,[1,1],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv0', trainable=False)
    net1=slim.conv2d(net0,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv1', trainable=False)
    
    net = tf.concat([net0,net1],axis=3)
    netaggi_0 = agg(net,scope='g_aggi_0')
    netaggm_0 = agg(net,scope='g_aggm_0')

    net1 = netaggi_0 * tf.nn.sigmoid(netaggm_0)

    net2=slim.conv2d(net1,channel,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv2', trainable=False)
    net3=slim.conv2d(net2,channel,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv3', trainable=False)

    #agg
    netaggi_1 = agg(tf.concat([netaggi_0,net3,net2],axis=3),scope='g_aggi_1')
    netaggm_1 = agg(tf.concat([netaggm_0,net3,net2],axis=3),scope='g_aggm_1')

    net3 = netaggi_1 * tf.nn.sigmoid(netaggm_1)

    net4=slim.conv2d(net3,channel,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv4', trainable=False)
    net5=slim.conv2d(net4,channel,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv5', trainable=False)
    
    #agg
    netaggi_2 = agg(tf.concat([netaggi_1,net5,net4],axis=3),scope='g_aggi_2')
    netaggm_2 = agg(tf.concat([netaggm_1,net5,net4],axis=3),scope='g_aggm_2')

    net6 = netaggi_2 * tf.nn.sigmoid(netaggm_2)


    net6=slim.conv2d(net6,channel,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv6', trainable=False)
    net7=slim.conv2d(net6,channel,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv7', trainable=False)

    #agg
    netimg = agg(tf.concat([netaggi_1,netaggi_2,net6,net7],axis=3),scope='g_aggi_3')
    netmask = agg(tf.concat([netaggm_1,netaggm_2,net6,net7],axis=3),scope='g_aggm_3')

    netimg = spp(netimg,scope='g_imgpool')
    netmask = spp(netmask,scope='g_maskpool')

    netimg = netimg * tf.nn.sigmoid(netmask)

    netimg=slim.conv2d(netimg,3,[1,1],rate=1,activation_fn=None,scope='g_conv_img', trainable=False)
    netmask=slim.conv2d(netmask,1,[1,1],rate=1,activation_fn=None,scope='g_conv_mask', trainable=False)

    return netimg,netmask



def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02), trainable=False)
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID", trainable=False)
        return conv

def lreluX(x, a):
    with tf.name_scope("lreluX"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02), trainable=False)
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False, trainable=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon, trainable=False, is_training=False)
        return normalized

def build_discriminator(discrim_inputs,discrim_targets):
    n_layers = 3
    layers = []
    channel=64

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv(input, channel, stride=2)
        rectified = lreluX(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = channel * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = conv(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lreluX(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1],layers


def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32), trainable=False)
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0), trainable=False)
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True, trainable=False)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        # return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
        #                     weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
        #                     biases_initializer=None)
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None, trainable=False)

def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        # return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
        #                             weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
        #                             biases_initializer=None)
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None, trainable=False)

def residule_block(x, dim, ks=3, s=1, name='res'):
    p = int((ks - 1) / 2)
    y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
    y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
    y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
    y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
    return y + x

def build_shadow_generator(input,channel=64):
    c0 = tf.pad(input, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
    c1 = tf.nn.relu(instance_norm(conv2d(c0, channel, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
    c2 = tf.nn.relu(instance_norm(conv2d(c1, channel*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
    c3 = tf.nn.relu(instance_norm(conv2d(c2, channel*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
    r1 = residule_block(c3, channel*4, name='g_r1')
    r2 = residule_block(r1, channel*4, name='g_r2')
    r3 = residule_block(r2, channel*4, name='g_r3')
    r4 = residule_block(r3, channel*4, name='g_r4')
    r5 = residule_block(r4, channel*4, name='g_r5')
    r6 = residule_block(r5, channel*4, name='g_r6')
    r7 = residule_block(r6, channel*4, name='g_r7')
    r8 = residule_block(r7, channel*4, name='g_r8')
    r9 = residule_block(r8, channel*4, name='g_r9')

    d1 = deconv2d(r9, channel*2, 3, 2, name='g_d1_dc')
    d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
    d2 = deconv2d(d1, channel, 3, 2, name='g_d2_dc')
    d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
    d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
    pred = conv2d(d2, 3, 7, 1, padding='VALID', name='g_pred_c')
    return pred