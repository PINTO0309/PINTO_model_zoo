import cv2
import numpy as np
import random

def pixel_jitter(src,p=0.5,max_=5.):

    src=src.astype(np.float32)

    pattern=(np.random.rand(src.shape[0], src.shape[1],src.shape[2])-0.5)*2*max_
    img = src + pattern

    img[img<0]=0
    img[img >255] = 255

    img = img.astype(np.uint8)

    return img

def gray(src):
    g_img=cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
    src[:,:,0]=g_img
    src[:,:,1]=g_img
    src[:,:,2]=g_img
    return src

def swap_change(src):
    a = [0,1,2]

    k = random.sample(a, 3)

    res=src.copy()
    res[:,:,0]=src[:,:,k[0]]
    res[:, :, 1] = src[:, :, k[1]]
    res[:, :, 2] = src[:, :, k[2]]
    return res


def Img_dropout(src,max_pattern_ratio=0.05):
    pattern=np.ones_like(src)
    width_ratio = random.uniform(0, max_pattern_ratio)
    height_ratio = random.uniform(0, max_pattern_ratio)
    width=src.shape[1]
    height=src.shape[0]
    block_width=width*width_ratio
    block_height=height*height_ratio
    width_start=int(random.uniform(0,width-block_width))
    width_end=int(width_start+block_width)
    height_start=int(random.uniform(0,height-block_height))
    height_end=int(height_start+block_height)
    pattern[height_start:height_end,width_start:width_end,:]=0
    img=src*pattern
    return img



def blur_heatmap(src, ksize=(3, 3)):
    for i in range(src.shape[2]):
        src[:, :, i] = cv2.GaussianBlur(src[:, :, i], ksize, 0)
        amin, amax = src[:, :, i].min(), src[:, :, i].max()  # 求最大最小值
        if amax>0:
            src[:, :, i] = (src[:, :, i] - amin) / (amax - amin)  # (矩阵元素-最小值)/(最大值-最小值)
    return src
def blur(src,ksize=(3,3)):
    for i in range(src.shape[2]):
        src[:, :, i]=cv2.GaussianBlur(src[:, :, i],ksize,1.5)
    return src




def adjust_contrast(image, factor):
    """ Adjust contrast of an image.
    Args
        image: Image to adjust.
        factor: A factor for adjusting contrast.
    """
    mean = image.mean(axis=0).mean(axis=0)
    return _clip((image - mean) * factor + mean)


def adjust_brightness(image, delta):
    """ Adjust brightness of an image
    Args
        image: Image to adjust.
        delta: Brightness offset between -1 and 1 added to the pixel values.
    """
    return _clip(image + delta * 255)


def adjust_hue(image, delta):
    """ Adjust hue of an image.
    Args
        image: Image to adjust.
        delta: An interval between -1 and 1 for the amount added to the hue channel.
               The values are rotated if they exceed 180.
    """
    image[..., 0] = np.mod(image[..., 0] + delta * 180, 180)
    return image


def adjust_saturation(image, factor):
    """ Adjust saturation of an image.
    Args
        image: Image to adjust.
        factor: An interval for the factor multiplying the saturation values of each pixel.
    """
    image[..., 1] = np.clip(image[..., 1] * factor, 0, 255)
    return image


def _clip(image):
    """
    Clip and convert an image to np.uint8.
    Args
        image: Image to clip.
    """
    return np.clip(image, 0, 255).astype(np.uint8)
def _uniform(val_range):
    """ Uniformly sample from the given range.
    Args
        val_range: A pair of lower and upper bound.
    """
    return np.random.uniform(val_range[0], val_range[1])


class ColorDistort():

    def __init__(
            self,
            contrast_range=(0.8, 1.2),
            brightness_range=(-.2, .2),
            hue_range=(-0.1, 0.1),
            saturation_range=(0.8, 1.2)
    ):
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.hue_range = hue_range
        self.saturation_range = saturation_range

    def __call__(self, image):


        if self.contrast_range is not None:
            contrast_factor = _uniform(self.contrast_range)
            image = adjust_contrast(image,contrast_factor)
        if self.brightness_range is not None:
            brightness_delta = _uniform(self.brightness_range)
            image = adjust_brightness(image, brightness_delta)

        if self.hue_range is not None or self.saturation_range is not None:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            if self.hue_range is not None:
                hue_delta = _uniform(self.hue_range)
                image = adjust_hue(image, hue_delta)

            if self.saturation_range is not None:
                saturation_factor = _uniform(self.saturation_range)
                image = adjust_saturation(image, saturation_factor)

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image




class DsfdVisualAug():
    pass