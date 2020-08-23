import tensorflow as tf

def batch_norm():

    return tf.keras.layers.BatchNormalization(fused=True,
                                              momentum=0.997,
                                              epsilon=1e-5)


def channel_shuffle(z):

    shape = tf.shape(z)
    batch_size = shape[0]
    height, width = shape[1], shape[2]

    depth = z.shape[3]

    z = tf.reshape(z, [batch_size, height, width, 2,depth//2])  # shape [batch_size, height, width, 2, depth]

    z = tf.transpose(z, [0, 1, 2, 4, 3])
    z = tf.reshape(z, [batch_size, height, width, depth])
    x, y = tf.split(z, num_or_size_splits=2, axis=3)
    return x, y

class HS(tf.keras.Model):

    def __init__(self):
        super(HS, self).__init__()

    def call(self, inputs):

        x = inputs*tf.nn.relu6(inputs+3.)/6.

        return x

class SELayer(tf.keras.Model):

    def __init__(self,
                 inplanes,
                 kernel_initializer='glorot_normal'):
        super(SELayer, self).__init__()

        self.pool1=tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')

        self.conv1=tf.keras.layers.Conv2D(inplanes//4,
                                          kernel_size=[1,1],
                                          strides=1,
                                          use_bias=False,
                                          kernel_initializer=kernel_initializer)
        self.bn1=batch_norm()
        self.conv2=tf.keras.layers.Conv2D(inplanes,
                                          kernel_size=[1,1],
                                          strides=1,
                                          use_bias=False,
                                          kernel_initializer=kernel_initializer)


        self.attention_act=HS()
    def call(self, inputs, training=False):

        se_pool=tf.expand_dims(self.pool1(inputs),axis=1)
        se_pool = tf.expand_dims(se_pool, axis=2)

        se_conv1=self.conv1(se_pool)
        se_bn1 = self.bn1(se_conv1,training=training)
        se_conv2 = self.conv2(se_bn1)

        attention=self.attention_act(se_conv2)

        outputs=inputs*attention
        return outputs

class Shufflenet(tf.keras.Model):
    def __init__(self,
                 inp,
                 oup,
                 base_mid_channels,
                 *,
                 ksize,
                 stride,
                 activation,
                 useSE,
                 kernel_initializer='glorot_normal'):
        super(Shufflenet, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]
        assert base_mid_channels == oup // 2

        self.base_mid_channel = base_mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
        ##pw
        tf.keras.layers.Conv2D(base_mid_channels,
                               kernel_size=[1,1],
                               strides=1,
                               padding='valid',
                               use_bias=False,
                               kernel_initializer=kernel_initializer),
        batch_norm(),
        None,
        ##dw
        tf.keras.layers.SeparableConv2D(base_mid_channels,
                                        kernel_size=[1, 1],
                                        strides=stride,
                                        padding='valid',
                                        depth_multiplier=1,
                                        use_bias=False,
                                        kernel_initializer=kernel_initializer),
        batch_norm(),

        ##pw
        tf.keras.layers.Conv2D(outputs,
                               kernel_size=[1, 1],
                               strides=1,
                               padding='valid',
                               use_bias=False,
                               kernel_initializer=kernel_initializer),
        batch_norm(),
        None,
        ]

        if activation == 'ReLU':
            assert useSE == False
            '''This model should not have SE with ReLU'''
            branch_main[2] = tf.keras.layers.ReLU()
            branch_main[-1] = tf.keras.layers.ReLU()
        else:
            branch_main[2] = HS()
            branch_main[-1] = HS()
            if useSE:
                branch_main.append(SELayer(outputs))
        self.branch_main = tf.keras.Sequential(branch_main)



        if stride == 2:
            branch_proj = [
                # dw
                tf.keras.layers.SeparableConv2D(inp,
                                                (ksize,ksize),
                                                stride,
                                                padding='same',
                                                depth_multiplier=1,
                                                use_bias=False,
                                                kernel_initializer=kernel_initializer),
                batch_norm(),
                # pw-linear
                tf.keras.layers.Conv2D(inp,
                                       (1,1),
                                       1,
                                       padding='valid',
                                       use_bias=False,
                                       kernel_initializer=kernel_initializer),
                batch_norm(),
                None,
            ]
            if activation == 'ReLU':
                branch_proj[-1] = tf.keras.layers.ReLU()
            else:
                branch_proj[-1] = HS()
            self.branch_proj = tf.keras.Sequential(branch_proj)
        else:
            self.branch_proj = None



    def call(self, inputs, training=False):


        if self.stride == 1:
            x_proj, x = channel_shuffle(inputs)

            return tf.concat((x_proj, self.branch_main(x,training=training)), 3)
        elif self.stride == 2:

            x_proj = inputs
            x = inputs

            return tf.concat((self.branch_proj(x_proj,training=training),
                              self.branch_main(x,training=training)), 3)


class Shuffle_Xception(tf.keras.Model):

    def __init__(self,
                 inp,
                 oup,
                 base_mid_channels,
                 *,
                 stride,
                 activation,
                 useSE,
                 kernel_initializer='glorot_normal'):
        super(Shuffle_Xception, self).__init__()

        assert stride in [1, 2]
        assert base_mid_channels == oup//2

        self.base_mid_channel = base_mid_channels
        self.stride = stride
        self.ksize = 3
        self.pad = 1
        self.inp = inp
        outputs = oup - inp

        branch_main = [
            # dw
            tf.keras.layers.SeparableConv2D(inp,
                                            kernel_size=[3,3],
                                            strides=stride,
                                            padding='same',
                                            use_bias=False,
                                            kernel_initializer=kernel_initializer),
            batch_norm(),
            # pw

            tf.keras.layers.Conv2D(base_mid_channels,
                                   (1, 1),
                                   1,
                                   padding='valid',
                                   use_bias=False,
                                   kernel_initializer=kernel_initializer),
            batch_norm(),
            None,
            # dw
            tf.keras.layers.SeparableConv2D(base_mid_channels,
                                            kernel_size=[3, 3],
                                            strides=stride,
                                            padding='same',
                                            use_bias=False,
                                            kernel_initializer=kernel_initializer),
            batch_norm(),
            # pw

            tf.keras.layers.Conv2D(base_mid_channels,
                                   (1, 1),
                                   1,
                                   padding='valid',
                                   use_bias=False,
                                   kernel_initializer=kernel_initializer),
            batch_norm(),
            None,
            # dw

            tf.keras.layers.SeparableConv2D(base_mid_channels,
                                            kernel_size=[3, 3],
                                            strides=stride,
                                            padding='same',
                                            use_bias=False,
                                            kernel_initializer=kernel_initializer),
            batch_norm(),
            # pw
            tf.keras.layers.Conv2D(base_mid_channels,
                                   (1, 1),
                                   1,
                                   padding='valid',
                                   use_bias=False,
                                   kernel_initializer=kernel_initializer),
            batch_norm(),
            None,
        ]

        if activation == 'ReLU':
            branch_main[4] = tf.keras.layers.ReLU()
            branch_main[9] = tf.keras.layers.ReLU()
            branch_main[14] = tf.keras.layers.ReLU()
        else:
            branch_main[4] = HS()
            branch_main[9] = HS()
            branch_main[14] = HS()
        assert None not in branch_main

        if useSE:
            assert activation != 'ReLU'
            branch_main.append(SELayer(outputs))

        self.branch_main = tf.keras.Sequential(branch_main)

        if self.stride == 2:
            branch_proj = [
                # dw

                tf.keras.layers.SeparableConv2D(inp,
                                                kernel_size=[3, 3],
                                                strides=stride,
                                                padding='same',
                                                use_bias=False,
                                                kernel_initializer=kernel_initializer),
                batch_norm(),
                # pw-linear
                tf.keras.layers.Conv2D(base_mid_channels,
                                       kernel_size=(1, 1),
                                       strides=1,
                                       padding='valid',
                                       use_bias=False,
                                       kernel_initializer=kernel_initializer),
                batch_norm(),
                None,
            ]
            if activation == 'ReLU':
                branch_proj[-1] = tf.keras.layers.ReLU()
            else:
                branch_proj[-1] = HS()
            self.branch_proj =  tf.keras.Sequential(branch_proj)

    def call(self, inputs, training=False):
        if self.stride==1:
            x_proj, x = channel_shuffle(inputs)
            return tf.concat((x_proj, self.branch_main(x,training=training)), 3)
        elif self.stride==2:
            x_proj = inputs
            x = inputs
            return tf.concat((self.branch_proj(x_proj,training=training),
                              self.branch_main(x,training=training)), 3)

class ShuffleNetPlus(tf.keras.Model):
    def __init__(self,
                 model_size='Small',
                 kernel_initializer='glorot_normal'):

        super(ShuffleNetPlus, self).__init__()

        architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]

        self.stage_repeats = [4, 4, 8, 4]


        if model_size == 'Large':
            self.stage_out_channels = [-1, 16, 68, 168, 336, 672, 1280]
        elif model_size == 'Medium':
            self.stage_out_channels = [-1, 16, 48, 128, 256, 512, 1280]
        elif model_size == 'Small':
            self.stage_out_channels = [-1, 16, 36, 104, 208, 416, 1280]

        # building first layer
        input_channel = self.stage_out_channels[1]

        self.first_conv = tf.keras.Sequential(

            [
            tf.keras.layers.Conv2D(input_channel,
                                   kernel_size=(3,3),
                                   strides=2,
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=kernel_initializer),
            batch_norm(),
            HS()]
        )



        self.features = []
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            activation = 'HS' if idxstage >= 1 else 'ReLU'
            useSE = 'True' if idxstage >= 2 else False

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                blockIndex = architecture[archIndex]
                archIndex += 1
                if blockIndex == 0:
                    print('Shuffle3x3')
                    self.features.append(Shufflenet(inp,
                                                    outp,
                                                    base_mid_channels=outp // 2,
                                                    ksize=3,
                                                    stride=stride,
                                                    activation=activation,
                                                    useSE=useSE,
                                                    kernel_initializer=kernel_initializer))
                elif blockIndex == 1:
                    print('Shuffle5x5')
                    self.features.append(Shufflenet(inp,
                                                    outp,
                                                    base_mid_channels=outp // 2,
                                                    ksize=5,
                                                    stride=stride,
                                                    activation=activation,
                                                    useSE=useSE,
                                                    kernel_initializer=kernel_initializer))
                elif blockIndex == 2:
                    print('Shuffle7x7')
                    self.features.append(Shufflenet(inp,
                                                    outp,
                                                    base_mid_channels=outp // 2,
                                                    ksize=7,
                                                    stride=stride,
                                                    activation=activation,
                                                    useSE=useSE,
                                                    kernel_initializer=kernel_initializer))
                elif blockIndex == 3:
                    print('Xception')
                    self.features.append(Shuffle_Xception(inp,
                                                          outp,
                                                          base_mid_channels=outp // 2,
                                                          stride=stride,
                                                          activation=activation,
                                                          useSE=useSE,
                                                          kernel_initializer=kernel_initializer))
                else:
                    raise NotImplementedError
                input_channel = output_channel
        assert archIndex == len(architecture)



    def call(self, inputs, training=False):

        x = self.first_conv(inputs,training=training)

        endpoints = {}

        for i,stage in enumerate(self.features):

            x=stage(x,training=training)
            endpoints['layer%d'%(i+1)] = x

        return x,endpoints



if __name__=='__main__':

    import numpy as np

    model = ShuffleNetPlus()

    image=np.zeros(shape=(1,160,160,3),dtype=np.float32)

    x=model.predict(image)


    model.summary()
