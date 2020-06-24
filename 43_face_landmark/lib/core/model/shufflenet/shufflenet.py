import tensorflow as tf

def batch_norm():
    return tf.keras.layers.BatchNormalization(fused=True,
                                              momentum=0.997,
                                              epsilon=1e-5)

def concat_shuffle_split(x,y):
    z = tf.concat([x, y], axis=3)
    shape = tf.shape(z)
    batch_size = shape[0]
    height, width = shape[1], shape[2]

    depth = z.shape[3]

    z = tf.reshape(z, [batch_size, height, width, 2,depth//2])  # shape [batch_size, height, width, 2, depth]

    z = tf.transpose(z, [0, 1, 2, 4, 3])
    z = tf.reshape(z, [batch_size, height, width, depth])
    x, y = tf.split(z, num_or_size_splits=2, axis=3)
    return x, y

def concat_shuffle_split_abstract(x, y):
    x_cp_1 = x[:, :, :, 0::2]
    y_cp_1 = y[:, :, :, 0::2]
    x=tf.concat([x_cp_1, y_cp_1], axis=3)


    x_cp_2 = x[:, :, :, 1::2]
    y_cp_2 = y[:, :, :, 1::2]
    y = tf.concat([x_cp_2, y_cp_2], axis=3)

    return x, y

class basic_unit(tf.keras.Model):
    def __init__(self,
                 output_size,
                 kernel_initializer='glorot_normal'):
        super(basic_unit, self).__init__()


        self.basic_unit = tf.keras.Sequential(

                    [
                        tf.keras.layers.Conv2D(output_size,
                                               kernel_size=(1, 1),
                                               strides=1,
                                               padding='same',
                                               use_bias=False,
                                               kernel_initializer=kernel_initializer),
                        batch_norm(),
                        tf.keras.layers.ReLU(),

                        tf.keras.layers.SeparableConv2D(output_size,
                                                        kernel_size=(3, 3),
                                                        strides=1,
                                                        padding='same',
                                                        use_bias=False,
                                                        kernel_initializer=kernel_initializer),
                        batch_norm(),
                        tf.keras.layers.ReLU()
                    ]
                    )

    def call(self, inputs, training=False):

        x=self.basic_unit(inputs,training=training)
        return x

class basic_unit_with_downsampling(tf.keras.Model):
    def __init__(self,
                 output_size,
                 kernel_initializer='glorot_normal'):
        super(basic_unit_with_downsampling, self).__init__()

        self.basic_unit_branch = tf.keras.Sequential(

            [
                tf.keras.layers.Conv2D(output_size,
                                       kernel_size=(1, 1),
                                       strides=1,
                                       padding='same',
                                       use_bias=False,
                                       kernel_initializer=kernel_initializer),
                batch_norm(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.SeparableConv2D(output_size,
                                                kernel_size=(3, 3),
                                                strides=2,
                                                padding='same',
                                                use_bias=False,
                                                kernel_initializer=kernel_initializer),
                batch_norm(),
                tf.keras.layers.ReLU()
            ]
        )

        self.project_branch = tf.keras.Sequential(

            [tf.keras.layers.SeparableConv2D(output_size,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding='same',
                                            use_bias=False,
                                            kernel_initializer=kernel_initializer),
            batch_norm(),
            tf.keras.layers.ReLU()]
        )


    def call(self, inputs, training=False):
        x=self.basic_unit_branch(inputs,training=training)
        y=self.project_branch(inputs,training=training)

        return x,y


class ShufflenetBlock(tf.keras.Model):
    def __init__(self,
                 output_size,
                 repeat=4,
                 kernel_initializer='glorot_normal'):
        super(ShufflenetBlock, self).__init__()


        self.down_sample=basic_unit_with_downsampling(output_size//2,
                                                      kernel_initializer=kernel_initializer)

        self.basic_units=[basic_unit(output_size//2,
                                         kernel_initializer=kernel_initializer
                                         ) for i in range(2, repeat + 1)]


    def call(self, inputs, training=False):

        x,y=self.down_sample(inputs,training=training)

        for uint in self.basic_units:
            x, y = concat_shuffle_split_abstract(x, y)
            x = uint(x)
        x = tf.concat([x, y], axis=3)

        return x


class Shufflenet(tf.keras.Model):
    def __init__(self,
                 model_size=1.0,
                 kernel_initializer='glorot_normal'):
        super(Shufflenet, self).__init__()

        possibilities = {'0.5': 48, '0.75': 96, '1.0': 116, '1.5': 176, '2.0': 224}
        self.initial_depth = possibilities[model_size]

        ### stride eual to 4
        self.first_conv = tf.keras.Sequential(

            [
                tf.keras.layers.Conv2D(24,
                                       kernel_size=(3, 3),
                                       strides=2,
                                       padding='same',
                                       use_bias=False,
                                       kernel_initializer=kernel_initializer),
                batch_norm(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                             strides=2)
                # tf.keras.layers.SeparableConv2D(32,
                #                                kernel_size=(3, 3),
                #                                strides=2,
                #                                padding='same',
                #                                use_bias=False,
                #                                kernel_initializer=kernel_initializer),
                # batch_norm(),
                # tf.keras.layers.ReLU()
            ]
            )


        self.block1=ShufflenetBlock(self.initial_depth,
                                     repeat=4,
                                     kernel_initializer=kernel_initializer)
        self.block2=ShufflenetBlock(self.initial_depth*2,
                                     repeat=8,
                                     kernel_initializer=kernel_initializer)
        self.block3 = ShufflenetBlock(self.initial_depth * 2 *2,
                                      repeat=4,
                                      kernel_initializer=kernel_initializer)


    def call(self, inputs, training=False):


        x=self.first_conv(inputs,training=training)



        x1=self.block1(x,training=training)


        x2=self.block2(x1, training=training)


        x3=self.block3(x2, training=training)



        return x1,x2,x3



if __name__=='__main__':

    import numpy as np

    model = Shufflenet()

    image=np.zeros(shape=(1,160,160,3),dtype=np.float32)

    x=model.predict(image)


    model.summary()



