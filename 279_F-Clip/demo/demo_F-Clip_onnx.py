#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import copy
import time
import argparse
import cv2 as cv
import numpy as np
import onnxruntime


mean = [109.730, 103.832, 98.681]
stddev = [22.275, 22.124, 23.229]

def run_inference(onnx_session, input_size, image):
    # Pre process
    org_H, org_W = image.shape[0], image.shape[1]
    H, W = input_size
    input_image = copy.deepcopy(image)
    output_image = copy.deepcopy(image)

    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = cv.resize(input_image, dsize=(W, H))
    input_image = (input_image - mean) / stddev
    input_image = input_image.transpose(2, 0, 1)
    input_image = input_image[np.newaxis, ...]
    input_image = input_image.astype(np.float32)

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    result = onnx_session.run(None, {input_name: input_image})
    # Post process
    lines = np.squeeze(result[0])
    score = np.squeeze(result[1])
    lines = lines[score > 0.4]
    lines = lines * 4
    lines[:, :, 0] = lines[:, :, 0] * org_H / H
    lines[:, :, 1] = lines[:, :, 1] * org_W / W

    for i in range(lines.shape[0]):
        start_coor = (int(lines[i][0][1]), int(lines[i][0][0]))
        end_coor = (int(lines[i][1][1]), int(lines[i][1][0]))
        cv.line(output_image, start_coor, end_coor, (0, 0, 255), 2, lineType=16)  # red

    return output_image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='fclip_hr_512x512/fclip_hr_512x512.onnx',
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default='512,512',
    )
    parser.add_argument(
        "--provider",
        choices=[
            'gpu',
            'openvino',
            'cpu'
        ],
        default='gpu',
    )

    args = parser.parse_args()
    model_path = args.model
    input_size = args.input_size
    input_size = [int(i) for i in input_size.split(',')]
    provider = args.provider

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
    window_name = 'F-Clip test'
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

    # Load model
    session_option = onnxruntime.SessionOptions()
    session_option.log_severity_level = 4

    if provider == 'gpu':
        onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=[
                (
                    'TensorrtExecutionProvider', {
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': os.path.dirname(model_path),
                        'trt_fp16_enable': True,
                    }
                ),
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ],
        )

    elif provider == 'openvino':
        session_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        onnxruntime.capi._pybind_state.set_openvino_device('CPU_FP32')

        onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=[
                'OpenVINOExecutionProvider',
                'CPUExecutionProvider'
            ],
        )
    elif provider == 'cpu':
        onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=[
                'CPUExecutionProvider'
            ],
        )


    # Warmup
    ret, frame = cap.read()
    if not ret:
        sys.exit(0)
    debug_image = copy.deepcopy(frame)
    frame_height, frame_width = frame.shape[0], frame.shape[1]
    _ = run_inference(
        onnx_session,
        input_size,
        frame,
    )
    start_time = time.time()
    _ = run_inference(
        onnx_session,
        input_size,
        frame,
    )
    elapsed_time = time.time() - start_time
    real_fps = int(1/elapsed_time)
    output_file = 'output.mp4'
    out = None
    if cap_fps < real_fps:
        out = cv.VideoWriter(output_file, fourcc, cap_fps, (w,h))
    else:
        out = cv.VideoWriter(output_file, fourcc, real_fps, (w,h))

    # Main
    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        start_time = time.time()

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
            output_image,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
            cv.LINE_AA
        )
        cv.putText(
            output_image,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            1,
            cv.LINE_AA
        )

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