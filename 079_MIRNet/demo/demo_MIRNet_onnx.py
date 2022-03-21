#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, input_size, image):
    # Pre process:Resize, BGR->RGB, Transpose, float32 cast
    input_image = cv.resize(image, dsize=(input_size[1], input_size[0]))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')
    input_image = input_image / 255.0

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    result = onnx_session.run(None, {input_name: input_image})

    # Post process:squeeze, RGB->BGR, Transpose, uint8 cast
    output_image = np.squeeze(result[0])
    # output_image = output_image.transpose(1, 2, 0)
    output_image = np.clip(output_image * 255.0, 0, 255)
    output_image = output_image.astype(np.uint8)
    output_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)

    return output_image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='saved_model_360x640/model_float32.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='360,640',
    )

    args = parser.parse_args()
    model_path = args.model
    input_size = args.input_size

    input_size = [int(i) for i in input_size.split(',')]

    # Initialize video capture
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie
    cap = cv.VideoCapture(cap_device)
    cap_fps = cap.get(cv.CAP_PROP_FPS)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) * 2
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv.VideoWriter('output.mp4', fourcc, cap_fps, (w,h))
    window_name = 'MIRNet test'
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=[
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': os.path.dirname(model_path),
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'OpenVINOExecutionProvider',
            'CPUExecutionProvider'
        ],
    )

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        # Inference execution
        output_image = run_inference(
            onnx_session,
            input_size,
            frame,
        )

        output_image = cv.resize(
            output_image,
            dsize=(frame_width, frame_height)
        )
        elapsed_time = time.time() - start_time

        # Inference elapsed time
        cv.putText(
            debug_image,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        combined_img = np.vstack([debug_image, output_image])
        cv.imshow(window_name, combined_img)
        out.write(combined_img)

    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()