#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(
    onnx_session,
    input_size,
    image,
    highpercent,
    lowpercent,
    hsvgamma,
    maxrange,
):
    # Pre process:Resize, BGR->RGB, Transpose, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')
    input_image = input_image / 255.0

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: input_image})

    # Save original image
    output = np.array(result)[0][0]
    output_original = copy.deepcopy(output)

    # Post process
    gray_output = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
    percent_max = sum(sum(gray_output >= maxrange)) / sum(
        sum(gray_output <= 1.0))
    max_value = np.percentile(gray_output[:], highpercent)
    if percent_max < (100 - highpercent) / 100.:
        scale = maxrange / max_value
        output = output * scale
        output = np.minimum(output, 1.0)

    sub_value = np.percentile(gray_output[:], lowpercent)
    output = ((output - sub_value) * (1. / (1 - sub_value)))

    hsv_image = cv.cvtColor(output, cv.COLOR_RGB2HSV)
    h_value, s_value, v_value = cv.split(hsv_image)
    s_value = np.power(s_value, hsvgamma)
    hsv_image = cv.merge((h_value, s_value, v_value))
    output = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

    output = np.minimum(output, 1.0)
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

    return output_original, output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=
        'saved_model_Syn_img_lowlight_withnoise_180x320/model_float32.onnx',
    )

    parser.add_argument(
        "--input_size",
        type=str,
        default='180,320',
    )
    parser.add_argument(
        "--highpercent",
        "-hp",
        type=int,
        default=95,
        help='should be in [85,100], linear amplification',
    )
    parser.add_argument(
        "--lowpercent",
        "-lp",
        type=int,
        default=5,
        help='should be in [0,15], rescale the range [p%,1] to [0, 1]',
    )
    parser.add_argument(
        "--gamma",
        "-g",
        type=int,
        default=8,
        help='should be in [6,10], increase the saturability',
    )
    parser.add_argument(
        "--maxrange",
        "-mr",
        type=int,
        default=8,
        help='linear amplification range',
    )

    args = parser.parse_args()
    model_path = args.model
    input_size = args.input_size

    input_size = [int(i) for i in input_size.split(',')]

    highpercent = args.highpercent
    lowpercent = args.lowpercent
    gamma = args.gamma / 10.0
    maxrange = args.maxrange / 10.0

    # Initialize video capture
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie
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
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        # Inference execution
        output_original, output_post_processed = run_inference(
            onnx_session,
            input_size,
            frame,
            highpercent,
            lowpercent,
            gamma,
            maxrange,
        )

        output_original = cv.resize(output_original,
                                    dsize=(frame_width, frame_height))
        output_post_processed = cv.resize(output_post_processed,
                                          dsize=(frame_width, frame_height))

        elapsed_time = time.time() - start_time

        # Inference elapsed time
        cv.putText(
            debug_image,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('MBLLEN Input', debug_image)
        cv.imshow('MBLLEN Output(model output)', output_original)
        cv.imshow('MBLLEN Output(post processed)', output_post_processed)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()