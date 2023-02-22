#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf


def run_inference(interpreter, image):
    # TFLite Infomation
    input_details = interpreter.get_input_details()
    input_size = input_details[0]['shape']
    input_width = input_size[2]
    input_height = input_size[1]

    # Pre process:Resize, BGR->RGB, Normarize, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image / 255.0
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    # Post process:convert numpy array
    output_details = interpreter.get_output_details()
    result = interpreter.get_tensor(output_details[0]['index'])

    # Post process
    depth_map = np.array(result)
    depth_map = np.squeeze(depth_map)
    depth_map = depth_map[:, :, 0]

    color_scaling = 1 / 64.0
    depth_map *= color_scaling
    depth_map = depth_map * 255.0
    depth_map = np.asarray(depth_map, dtype="uint8")

    return depth_map


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='pydnet2_192x512/model_float16_quant.tflite',
    )

    args = parser.parse_args()
    model_path = args.model

    # Initialize video capture
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie
    cap = cv.VideoCapture(cap_device)

    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Inference execution
        depth_map = run_inference(interpreter, frame)

        elapsed_time = time.time() - start_time

        # Draw
        debug_image, depth_image = draw_debug(
            debug_image,
            elapsed_time,
            depth_map,
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('PyDNet2 RGB Demo', debug_image)
        cv.imshow('PyDNet2 Depth Demo', depth_image)

    cap.release()
    cv.destroyAllWindows()


def draw_debug(image, elapsed_time, depth_map):
    image_width, image_height = image.shape[1], image.shape[0]
    debug_image = copy.deepcopy(image)

    # Apply ColorMap
    depth_image = cv.applyColorMap(depth_map, cv.COLORMAP_JET)
    depth_image = cv.resize(depth_image, dsize=(image_width, image_height))

    # Inference elapsed time
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
               cv.LINE_AA)

    return debug_image, depth_image


if __name__ == '__main__':
    main()
