#!/usr/bin/env python
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def run_inference(onnx_session, image):
    # ONNX Input Size
    input_size = onnx_session.get_inputs()[0].shape
    input_width = input_size[3]
    input_height = input_size[2]

    # Pre process:Resize, BGR->RGB, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input_image = (input_image / 255.0 - mean) / std
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: input_image})

    # Post process:squeeze, Sigmoid, Normarize, uint8 cast
    output_image = np.squeeze(result[0])
    output_image = sigmoid(output_image)
    min_value = np.min(output_image)
    max_value = np.max(output_image)
    output_image = (output_image - min_value) / (max_value - min_value)
    output_image *= 255
    output_image = output_image.astype('uint8')

    return output_image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='dgnet_s_480x640/dgnet_s_480x640.onnx',
    )

    args = parser.parse_args()
    model_path = args.model

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
        output_image = run_inference(
            onnx_session,
            frame,
        )

        elapsed_time = time.time() - start_time

        # Inference elapsed time
        elapsed_time_text = "Elapsed time: "
        elapsed_time_text += str(round((elapsed_time * 1000), 1))
        elapsed_time_text += 'ms'
        cv.putText(debug_image, elapsed_time_text, (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)

        # Map Resize
        output_image = cv.resize(
            output_image,
            dsize=(debug_image.shape[1], debug_image.shape[0]),
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('DGNet : Input', debug_image)
        cv.imshow('DGNet : Output', output_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
