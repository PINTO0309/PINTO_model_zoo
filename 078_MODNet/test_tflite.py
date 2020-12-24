import os
import cv2
import numpy as np
import time
from PIL import Image
try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

def resize_and_pad(img, size, pad_color=0):
    h, w = img.shape[:2]
    sh, sw = size
    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC
    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h
    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    # set pad color
    if len(img.shape) is 3 and not isinstance(pad_color, (list, tuple, np.ndarray)): # color image but only one color provided
        pad_color = [pad_color]*3
    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return scaled_img

img = Image.open('test2.jpg')
# img = img.resize((512, 512))
img = np.asarray(img)
# img = resize_and_pad(img, (512,512))
img = img / 255.
img = img.astype(np.float32)
img = img[np.newaxis,:,:,:]

# Tensorflow Lite
interpreter = Interpreter(model_path='modnet_512x512_float32.tflite', num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]['index']
output_details1 = interpreter.get_output_details()[0]['index']
output_details2 = interpreter.get_output_details()[1]['index']
output_details3 = interpreter.get_output_details()[2]['index']

interpreter.set_tensor(input_details, img)
interpreter.invoke()
matte = interpreter.get_tensor(output_details3)

print(matte.shape)
matte = matte[0]
matte = matte.reshape((512,512))
Image.fromarray(((matte * 255).astype('uint8')), mode='L').save('test2_matte.jpg')