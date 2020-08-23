from lib.dataset.dataietr import FaceKeypointDataIter
from train_config import config
import tensorflow as tf
import time
import argparse
import numpy as np
import os

import cv2

def vis_tflite(model):

    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_size = 160
    image_width = 640
    image_height = 480

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
    window_name = "USB Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    scalex = image_width / input_size
    scaley = image_height / input_size

    while True:
        start_time = time.perf_counter()

        ret, image = cam.read()
        if not ret:
            continue
        img_show = np.array(image)

        frame = cv2.resize(image, (input_size, input_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, axis=0)
        frame = frame.astype(np.float32)
        start=time.time()

        interpreter.set_tensor(input_details[0]['index'], frame)
        interpreter.invoke()
        tflite_res = interpreter.get_tensor(output_details[2]['index'])


        print('xxxx',time.time()-start)
        img_show=img_show.astype(np.uint8)
        landmark = np.array(tflite_res).reshape([-1, 2])

        for _index in range(landmark.shape[0]):
            x_y = landmark[_index]
            cv2.circle(img_show, center=(int(x_y[0] * input_size * scalex),
                                         int(x_y[1] * input_size * scaley)),
                       color=(255, 122, 122), radius=1, thickness=2)

        cv2.imshow('USB camera',img_show)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, default=None, help='the model to use')
    args = parser.parse_args()
    vis_tflite(args.model)




