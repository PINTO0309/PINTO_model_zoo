#!/usr/bin/env python
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def run_inference(onnx_session, image, score_th=None):
    # ONNX Input Size
    input_size = onnx_session.get_inputs()[0].shape
    input_width = input_size[3]
    input_height = input_size[2]

    # Pre process:Resize, BGR->RGB, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    mean = [0.5, 0.5, 0.5]
    std = [1.0, 1.0, 1.0]
    input_image = (input_image / 255.0 - mean) / std
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: input_image})

    # Post process:squeeze, Sigmoid, Normarize, uint8 cast
    mask = np.squeeze(result[0])
    min_value = np.min(mask)
    max_value = np.max(mask)
    mask = (mask - min_value) / (max_value - min_value)
    if score_th is not None:
        mask = np.where(mask < score_th, 0, 1)
    mask *= 255
    mask = mask.astype('uint8')

    return mask


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='isnet_480x640.onnx',
    )
    parser.add_argument("--score_th", type=float, default=None)

    args = parser.parse_args()
    model_path = args.model
    score_th = args.score_th

    # Initialize video capture
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie
    cap = cv.VideoCapture(cap_device)

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )

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
            frame,
            score_th,
        )

        elapsed_time = time.time() - start_time

        # Map Resize
        mask = cv.resize(
            mask,
            dsize=(debug_image.shape[1], debug_image.shape[0]),
            interpolation=cv.INTER_LINEAR,
        )

        # Mask Overlay
        overlay_image = np.zeros(debug_image.shape, dtype=np.uint8)
        overlay_image[:] = (255, 255, 255)
        mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
        mask_image = np.where(mask, debug_image, overlay_image)

        # Inference elapsed time
        elapsed_time_text = "Elapsed time: "
        elapsed_time_text += str(round((elapsed_time * 1000), 1))
        elapsed_time_text += 'ms'
        cv.putText(debug_image, elapsed_time_text, (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('IS-Net : Input', debug_image)
        cv.imshow('IS-Net : Output', mask)
        cv.imshow('IS-Net : Mask', mask_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
