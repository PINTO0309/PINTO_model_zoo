#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf


def run_inference(interpreter, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]

    # Pre process:Resize, expand dimensions, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image / 255.0
    input_image = input_image.astype('float32')
    input_image = np.expand_dims(input_image, axis=0)

    # Inference
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    result = interpreter.get_tensor(output_details[0]['index'])

    # Post process:squeeze, Transpose, Resize, argmax
    segmentation_map = result[0]
    segmentation_map = np.squeeze(segmentation_map)
    segmentation_map = cv.resize(
        segmentation_map,
        dsize=(image_width, image_height),
        interpolation=cv.INTER_LINEAR,
    )
    segmentation_map = segmentation_map.transpose(2, 0, 1)

    return segmentation_map


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_rsb_480x640/model_float16_quant.tflite',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='480,640',
    )

    args = parser.parse_args()

    model_path = args.model
    input_size = [int(i) for i in args.input_size.split(',')]

    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie

    # Initialize video capture
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
        segmentation_map = run_inference(
            interpreter,
            input_size,
            frame,
        )

        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            segmentation_map,
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('SUIM-Net Demo: Original', frame)
        cv.imshow('SUIM-Net Demo: Segmentation', debug_image)

    cap.release()
    cv.destroyAllWindows()


def get_color_map_list(num_classes, custom_color=None):
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3 + 2] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map


def draw_debug(image, elapsed_time, segmentation_map):
    # color map list
    color_map = get_color_map_list(5)

    # Overlay segmentation map
    for index, segmentation_mask in enumerate(segmentation_map):
        debug_image = copy.deepcopy(image)
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (color_map[index * 3 + 0], color_map[index * 3 + 1],
                       color_map[index * 3 + 2])

        # Overlay
        mask = np.where(segmentation_mask > 0.5, 0, 1)
        mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
        mask_image = np.where(mask, image, bg_image)
        image = cv.addWeighted(image, 0.25, mask_image, 0.75, 1.0)

    # Inference elapsed time
    cv.putText(image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
               cv.LINE_AA)

    return image


if __name__ == '__main__':
    main()
