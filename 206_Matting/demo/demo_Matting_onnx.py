#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, input_size, image):
    # Pre process:Resize, Standardization, Transpose, expand dimensions
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    input_image = (input_image / 255 - mean) / std
    input_image = input_image.transpose(2, 0, 1).astype('float32')
    input_image = np.expand_dims(input_image, axis=0)

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: input_image})

    # Post process:squeeze
    mask = result[0]
    mask = np.squeeze(mask)

    return mask


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument(
        "--model",
        type=str,
        default='modnet_hrnet_w18_256x256/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='256,256',
    )

    args = parser.parse_args()

    model_path = args.model
    input_size = [int(i) for i in args.input_size.split(',')]

    score = args.score

    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie

    # Initialize video capture
    cap = cv.VideoCapture(cap_device)

    # Load model
    onnx_session = onnxruntime.InferenceSession(model_path)

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Inference execution
        mask = run_inference(
            onnx_session,
            input_size,
            frame,
        )

        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            score,
            mask,
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('Matting Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_debug(image, elapsed_time, score, mask):
    image_width, image_height = image.shape[1], image.shape[0]

    # Match the size
    debug_image = copy.deepcopy(image)
    mask = cv.resize(
        mask,
        dsize=(image_width, image_height),
        interpolation=cv.INTER_LINEAR,
    )

    # overlay image
    overlay_image = np.zeros(image.shape, dtype=np.uint8)
    overlay_image[:] = (0, 255, 0)

    # Threshold check by score
    mask = np.where(mask > score, 0, 1)
    # Overlay segmentation map
    mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
    mask_image = np.where(mask, debug_image, overlay_image)
    debug_image = cv.addWeighted(debug_image, 0.5, mask_image, 0.5, 1.0)

    # Inference elapsed time
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
               cv.LINE_AA)

    return debug_image


if __name__ == '__main__':
    main()
