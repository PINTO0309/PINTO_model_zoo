import os
class config():
    def __init__(self):
        # Todo: set base_dir to kitti/image_2
        self.base_dir = '/media/b920405/Windows/datasets/kitti_dateset/'

        # Todo: set the base network: vgg16, vgg16_conv or mobilenet_v2
        self.network = 'mobilenet_v2'

        # set the bin size
        self.bin = 2

        # set the train/val split
        self.split = 0.8

        # set overlapping
        self.overlap = 0.1

        # set jittered
        self.jit = 3

        # set the normalized image size
        self.norm_w = 320
        self.norm_h = 320

        # set the batch size
        self.batch_size = 8

        # set the categories
        self.KITTI_cat = ['Car', 'Cyclist', 'Pedestrian']
        # self.KITTI_cat = ['Car']
