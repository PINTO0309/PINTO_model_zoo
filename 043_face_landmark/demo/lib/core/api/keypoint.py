#-*-coding:utf-8-*-
import tensorflow as tf
import cv2
import numpy as np
import time

from lib.helper.init import init
from train_config import config


class Keypoints:

    def __init__(self,pb):
        self.PADDING_FLAG=True
        self.ANCHOR_SIZE=0
        self._pb_path=pb

        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)

        with self._graph.as_default():

            self._graph, self._sess = init(self._pb_path)
            self.img_input = tf.get_default_graph().get_tensor_by_name('tower_0/images:0')
            self.embeddings_keypoints = tf.get_default_graph().get_tensor_by_name('tower_0/prediction:0')
            self.training = tf.get_default_graph().get_tensor_by_name('training_flag:0')


    def simple_run(self,img):

        with self._graph.as_default():
            img=np.expand_dims(img,axis=0)

            star=time.time()
            for i in range(10):
                [landmarkers] = self._sess.run([self.embeddings_keypoints], \
                                                        feed_dict={self.img_input: img, \
                                                               self.training: False})
            print((time.time()-star)/10)
        return landmarkers
    def run(self,_img,_bboxs):
        img_batch = []
        coordinate_correct_batch = []
        pad_batch = []

        res_landmarks_batch=[]
        res_state_batch=[]
        with self._sess.as_default():
            with self._graph.as_default():
                for _box in _bboxs:
                    crop_img=self._one_shot_preprocess(_img,_box)

                    landmarkers = self._sess.run([self.embeddings_keypoints], \
                                                        feed_dict={self.img_input: crop_img, \
                                                                   self.training: False})




    def _one_shot_preprocess(self,image,bbox):


        add = int(np.max(bbox))
        bimg = cv2.copyMakeBorder(image, add, add, add, add, borderType=cv2.BORDER_CONSTANT,
                                  value=config.DATA.pixel_means.tolist())

        bbox += add


        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        ###extend
        bbox[0] -= config.DATA.base_extend_range[0] * bbox_width
        bbox[1] -= config.DATA.base_extend_range[1] * bbox_height
        bbox[2] += config.DATA.base_extend_range[0] * bbox_width
        bbox[3] += config.DATA.base_extend_range[1] * bbox_height


        ##
        bbox = bbox.astype(np.int)
        crop_image = bimg[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        crop_image = cv2.resize(crop_image, (config.MODEL.hin, config.MODEL.win),
                                  interpolation=cv2.INTER_AREA)
        return crop_image