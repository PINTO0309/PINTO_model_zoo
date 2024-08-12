#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse
from typing import Any, Tuple

import cv2
import numpy as np
import onnxruntime  # type: ignore


def run_inference(
    onnx_session: onnxruntime.InferenceSession,
    image: np.ndarray,
) -> np.ndarray:
    input_detail = onnx_session.get_inputs()[0]
    input_name: str = input_detail.name
    input_shape: Tuple[int, int] = input_detail.shape[2:4]

    # Pre process: Resize, BGR->RGB, Transpose, float32 cast
    input_image: np.ndarray = cv2.resize(
        image,
        dsize=(input_shape[1], input_shape[0]),
    )
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')
    input_image = input_image / 255.0

    # Inference
    result: Any = onnx_session.run(None, {input_name: input_image})

    # Post process: squeeze, RGB->BGR, Transpose, uint8 cast
    output_image: np.ndarray = np.squeeze(result[0])
    output_image = output_image.transpose(1, 2, 0)
    output_image = output_image * 255
    output_image = output_image.astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    return output_image


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='diffusion_low_light_1x3x192x320.onnx',
    )

    args = parser.parse_args()
    model_path: str = args.model

    # Initialize video capture
    cap_device: Any = args.device
    if args.movie is not None:
        cap_device = args.movie
    cap: cv2.VideoCapture = cv2.VideoCapture(cap_device)

    # Load model
    onnx_session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
        model_path,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )

    while True:
        start_time: float = time.time()

        # Capture read
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        if not ret:
            break
        debug_image: np.ndarray = copy.deepcopy(frame)
        frame_height: int
        frame_width: int
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        # Inference execution
        output_image: np.ndarray = run_inference(
            onnx_session,
            frame,
        )

        output_image = cv2.resize(output_image,
                                  dsize=(frame_width, frame_height))

        elapsed_time: float = time.time() - start_time

        # Inference elapsed time
        cv2.putText(
            debug_image,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1,
            cv2.LINE_AA)

        key: int = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        cv2.imshow('Diffusion Low Light Input', debug_image)
        cv2.imshow('Diffusion Low Light Output', output_image)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
