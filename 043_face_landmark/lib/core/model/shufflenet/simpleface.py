# -*-coding:utf-8-*-
import sys
sys.path.append('.')
import tensorflow as tf

import math
import numpy as np

from train_config import config as cfg

from lib.core.model.shufflenet.shufflenet import Shufflenet
class SimpleFaceHead(tf.keras.Model):
    def __init__(self,
                 output_size,
                 kernel_initializer='glorot_normal'):
        super(SimpleFaceHead, self).__init__()

        self.output_size=output_size

        self.dense=tf.keras.layers.Dense(self.output_size,
                                         use_bias=True,
                                         kernel_initializer=kernel_initializer )

    def call(self, inputs):

        output=self.dense(inputs)

        return output


class SimpleFace(tf.keras.Model):

    def __init__(self,kernel_initializer='glorot_normal'):
        super(SimpleFace, self).__init__()

        model_size=cfg.MODEL.net_structure.split('_',1)[-1]
        self.backbone = Shufflenet(model_size=model_size,
                                   kernel_initializer=kernel_initializer)

        self.head=SimpleFaceHead(output_size=cfg.MODEL.out_channel,
                                 kernel_initializer=kernel_initializer)

        self.pool1 = tf.keras.layers.GlobalAveragePooling2D()
        self.pool2 = tf.keras.layers.GlobalAveragePooling2D()
        self.pool3 = tf.keras.layers.GlobalAveragePooling2D()


    @tf.function
    def call(self, inputs, training=False):
        inputs=self.preprocess(inputs)
        x1, x2, x3 = self.backbone(inputs, training=training)

        s1 = self.pool1(x1)
        s2 = self.pool2(x2)
        s3 = self.pool3(x3)

        multi_scale = tf.concat([s1, s2, s3], 1)

        out_put=self.head(multi_scale,training=training)

        return out_put



    @tf.function(input_signature=[tf.TensorSpec([None,cfg.MODEL.hin,cfg.MODEL.win,3], tf.float32)])
    def inference(self,images):
        inputs = self.preprocess(images)
        x1,x2,x3 = self.backbone(inputs, training=False)

        s1 = self.pool1(x1)
        s2 = self.pool2(x2)
        s3 = self.pool3(x3)

        multi_scale = tf.concat([s1, s2, s3], 1)

        out_put = self.head(multi_scale, training=False)


        landmark=out_put[:,:136]
        headpose=out_put[:,136:139]
        cls=out_put[:,139:]

        res={'landmark':landmark,
             'headpose':headpose,
             'cls':cls}

        return res



    def preprocess(self,image):

        mean = cfg.DATA.PIXEL_MEAN
        std =  cfg.DATA.PIXEL_STD

        image_mean = tf.constant(mean, dtype=tf.float32)
        image_invstd = tf.constant(std, dtype=tf.float32)
        image = (image - image_mean) / image_invstd

        return image




if __name__=='__main__':


    import time
    model = SimpleFace()

    image=np.zeros(shape=(1,160,160,3),dtype=np.float32)
    x=model.inference(image)
    tf.saved_model.save(model,'./model/keypoints')
    start=time.time()
    for i in range(100):
        x = model.inference(image)

    print('xxxyyyy',(time.time()-start)/100.)








def _wing_loss(landmarks, labels, w=10.0, epsilon=2.0,weights=1.):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, landmarks].  landmarks means x1,x2,x3,x4...y1,y2,y3,y4   1-D
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """

    x = landmarks - labels
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    absolute_x = tf.abs(x)
    losses = tf.where(
        tf.greater(w, absolute_x),
        w * tf.math.log(1.0 + absolute_x / epsilon),
        absolute_x - c
    )
    losses=losses*cfg.DATA.weights
    loss = tf.reduce_sum(tf.reduce_mean(losses*weights, axis=[0]))

    return loss

def _mse(landmarks, labels,weights=1.):

    return tf.reduce_mean(0.5*tf.square(landmarks - labels)*weights)

def l1(landmarks, labels):
    return tf.reduce_mean(landmarks - labels)

def calculate_loss(predict_keypoints, label_keypoints):
    

    landmark_label =      label_keypoints[:, 0:136]
    pose_label =          label_keypoints[:, 136:139]
    leye_cls_label =      label_keypoints[:, 139]
    reye_cls_label =      label_keypoints[:, 140]
    mouth_cls_label =     label_keypoints[:, 141]
    big_mouth_cls_label = label_keypoints[:, 142]


    landmark_predict =     predict_keypoints[:, 0:136]
    pose_predict =         predict_keypoints[:, 136:139]
    leye_cls_predict =     predict_keypoints[:, 139]
    reye_cls_predict =     predict_keypoints[:, 140]
    mouth_cls_predict =     predict_keypoints[:, 141]
    big_mouth_cls_predict = predict_keypoints[:, 142]











    loss = _wing_loss(landmark_predict, landmark_label)

    loss_pose = _mse(pose_predict, pose_label)



    leye_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=leye_cls_predict,
                                                                      labels=leye_cls_label) )
    reye_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=reye_cls_predict,
                                                                      labels=reye_cls_label))
    mouth_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mouth_cls_predict,
                                                                       labels=mouth_cls_label))
    mouth_loss_big = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=big_mouth_cls_predict,
                                                                        labels=big_mouth_cls_label))
    mouth_loss=mouth_loss+mouth_loss_big





    # ##make crosssentropy
    # leye_cla_correct_prediction = tf.equal(
    #     tf.cast(tf.greater_equal(tf.nn.sigmoid(leye_cls_predict), 0.5), tf.int32),
    #     tf.cast(leye_cla_label, tf.int32))
    # leye_cla_accuracy = tf.reduce_mean(tf.cast(leye_cla_correct_prediction, tf.float32))
    #
    # reye_cla_correct_prediction = tf.equal(
    #     tf.cast(tf.greater_equal(tf.nn.sigmoid(reye_cla_predict), 0.5), tf.int32),
    #     tf.cast(reye_cla_label, tf.int32))
    # reye_cla_accuracy = tf.reduce_mean(tf.cast(reye_cla_correct_prediction, tf.float32))
    # mouth_cla_correct_prediction = tf.equal(
    #     tf.cast(tf.greater_equal(tf.nn.sigmoid(mouth_cla_predict), 0.5), tf.int32),
    #     tf.cast(mouth_cla_label, tf.int32))
    # mouth_cla_accuracy = tf.reduce_mean(tf.cast(mouth_cla_correct_prediction, tf.float32))



    #### l2 regularization_losses
    # l2_loss = []
    # l2_reg = tf.keras.regularizers.l2(cfg.TRAIN.weight_decay_factor)
    # variables_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
    # for var in variables_restore:
    #     if 'weight' in var.name:
    #         l2_loss.append(l2_reg(var))
    # regularization_losses = tf.add_n(l2_loss, name='l1_loss')


    return loss+loss_pose+leye_loss+reye_loss+mouth_loss


