import os
import cv2
import numpy as np
import time
try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter


fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0


if __name__ == '__main__':

    camera_index  = 0
    camera_width  = 416
    camera_height = 416

    model_scale = 320

    # Tensorflow Lite
    interpreter = Interpreter(model_path='u2netp_{}x{}_float32.tflite'.format(model_scale, model_scale), num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]['index']
    output_details = interpreter.get_output_details()[0]['index']
