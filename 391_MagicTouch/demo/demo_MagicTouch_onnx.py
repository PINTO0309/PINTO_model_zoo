#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, image, prior_point):
    image_height, image_width = image.shape[0], image.shape[1]

    # ONNX Input Size
    input_size = onnx_session.get_inputs()[0].shape
    input_width = input_size[3]
    input_height = input_size[2]

    # Create Prior Map
    prior_map = np.zeros((image_height, image_width), np.float32)
    cv.circle(
        prior_map,
        center=prior_point,
        radius=5,
        color=(255, 255, 255),
        thickness=-1,
        lineType=cv.LINE_AA,
    )
    prior_map = cv.resize(prior_map, dsize=(input_width, input_height))

    # Pre process:Resize, BGR->RGB, Transpose, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image_b, input_image_g, input_image_r = cv.split(input_image)
    input_image = np.stack(
        [input_image_r, input_image_g, input_image_b, prior_map],
        axis=2,
    )
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')
    input_image = input_image / 255.0

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    result = onnx_session.run(None, {input_name: input_image})

    # Post process:squeeze
    result = np.array(result)
    output_image = np.squeeze(result)

    return output_image


def mouse_callback(event, x, y, flags, param):
    global mouse_point
    mouse_point = [x, y]


def main():
    global mouse_point

    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, default='sample.jpg')
    parser.add_argument(
        "--model",
        type=str,
        default='magic_touch.onnx',
    )

    args = parser.parse_args()
    model_path = args.model
    image_path = args.image

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )

    # Create GUI
    mouse_point = None
    cv.namedWindow('MagicTouch Input')
    cv.setMouseCallback('MagicTouch Input', mouse_callback)

    # Load Image
    image = cv.imread(image_path)

    while True:
        start_time = time.time()

        original_image = copy.deepcopy(image)
        debug_image = copy.deepcopy(image)
        image_height, image_width = image.shape[0], image.shape[1]

        # Inference execution
        result = run_inference(
            onnx_session,
            image,
            mouse_point,
        )

        result = cv.resize(
            result,
            dsize=(image_width, image_height),
        )

        elapsed_time = time.time() - start_time

        # Crop with mask
        black_image = np.zeros(debug_image.shape, dtype=np.uint8)
        mask = np.where(result > 0.5, 1.0, 0.0)
        mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
        debug_image = np.where(mask, debug_image, black_image)

        # Inference elapsed time
        cv.putText(
            original_image,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('MagicTouch Input', original_image)
        cv.imshow('MagicTouch Output', result)
        cv.imshow('MagicTouch Post-Process Image', debug_image)

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
