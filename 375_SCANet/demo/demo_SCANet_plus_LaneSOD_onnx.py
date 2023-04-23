#!/usr/bin/env python

import os
import copy
import time
import argparse
from typing import List
import cv2 as cv
import numpy as np
import onnxruntime


def run_inference_scanet(
    onnx_session: onnxruntime.InferenceSession,
    input_name: str,
    input_shape: List[int],
    image: np.ndarray,
) -> np.ndarray:
    # Pre process:Resize, BGR->RGB, Transpose, float32 cast
    input_image = cv.resize(
        src=image,
        dsize=(input_shape[3], input_shape[2]),
    )
    input_image = cv.cvtColor(
        src=input_image,
        code=cv.COLOR_BGR2RGB,
    )
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')
    input_image = input_image / 255.0
    input_image = (input_image - 0.5) / 0.5

    # Inference
    result = onnx_session.run(
        output_names=None,
        input_feed={input_name: input_image},
    )

    # Post process:squeeze, RGB->BGR, Transpose, uint8 cast
    output_image = np.squeeze(result[0])
    output_image = output_image.transpose(1, 2, 0)
    output_image = output_image * 0.5 + 0.5
    output_image = output_image * 255
    output_image = np.clip(output_image, 0, 255)
    output_image = output_image.astype(np.uint8)
    output_image = output_image[..., ::-1]

    return output_image


MEAN = np.asarray([[[[0.485]],[[0.456]],[[0.406]]]], dtype=np.float32)
STD = np.asarray([[[[0.229]],[[0.224]],[[0.225]]]], dtype=np.float32)

def run_inference_lanesod(
    onnx_session: onnxruntime.InferenceSession,
    input_name: str,
    input_shape: List[int],
    image: np.ndarray,
    threshold: float=0.05,
) -> np.ndarray:
    # Pre process:Resize, BGR->RGB, Transpose, float32 cast
    input_image = cv.resize(
        src=image,
        dsize=(input_shape[3], input_shape[2]),
    )
    input_image = cv.cvtColor(
        src=input_image,
        code=cv.COLOR_BGR2RGB,
    )
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')
    input_image = input_image / 255.0
    input_image = (input_image - MEAN) / STD

    # Inference
    result = onnx_session.run(
        output_names=None,
        input_feed={
            input_name: input_image,
        },
    )

    # Post process: Transpose, Resize, uint8 cast
    mask = result[0][0].transpose(1,2,0)
    mask = np.concatenate([mask, mask, mask], axis=2)
    mask = cv.resize(
        src=mask,
        dsize=(image.shape[1], image.shape[0]),
    )
    mask = np.where(mask >= threshold, 0, 1).astype(np.uint8)
    output_image = image * mask
    output_image[..., 1] = \
        np.where(
            output_image[..., 1] == 0, 255, output_image[..., 1]
        ).astype(np.uint8)

    return output_image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d',
        '--device',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-f',
        '--movie_file',
        type=str,
        default='360x640_dehaze.mp4',
    )
    parser.add_argument(
        '-ms',
        '--model_scanet',
        type=str,
        default='./scanet_gmodel_238_19.6429_0.7299_360x640.onnx',
    )
    parser.add_argument(
        '-ml',
        '--model_lanesod',
        type=str,
        default='./lanesod_384x640.onnx',
    )

    args = parser.parse_args()
    model_path_scanet = args.model_scanet
    model_path_lanesod = args.model_lanesod

    # Initialize video capture
    cap_device = args.device
    if args.movie_file is not None:
        cap_device = args.movie_file
    cap = cv.VideoCapture(cap_device)
    cap_fps = cap.get(cv.CAP_PROP_FPS)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(w, h*2),
    )
    WINDOW_NAME = 'SCANet test'
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)

    # Load model
    model_dir_scanet = os.path.dirname(model_path_scanet)
    if model_dir_scanet == '':
        model_dir_scanet = '.'
    model_dir_lanesod = os.path.dirname(model_path_lanesod)
    if model_dir_lanesod == '':
        model_dir_lanesod = '.'

    onnx_session_scanet = onnxruntime.InferenceSession(
        model_path_scanet,
        providers=[
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': model_dir_scanet,
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ],
    )
    onnx_session_lanesod = onnxruntime.InferenceSession(
        model_path_lanesod,
        providers=[
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': model_dir_lanesod,
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ],
    )

    model_input_scanet = onnx_session_scanet.get_inputs()[0]
    input_name_scanet = model_input_scanet.name
    input_shape_scanet = model_input_scanet.shape

    model_input_lanesod = onnx_session_lanesod.get_inputs()[0]
    input_name_lanesod = model_input_lanesod.name
    input_shape_lanesod = model_input_lanesod.shape


    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Inference execution
        start_time = time.time()

        output_image = run_inference_scanet(
            onnx_session=onnx_session_scanet,
            input_name=input_name_scanet,
            input_shape=input_shape_scanet,
            image=frame,
        )
        output_dehaze_image = run_inference_lanesod(
            onnx_session=onnx_session_lanesod,
            input_name=input_name_lanesod,
            input_shape=input_shape_lanesod,
            image=output_image,
        )

        elapsed_time = time.time() - start_time

        output_nodehaze_image = run_inference_lanesod(
            onnx_session=onnx_session_lanesod,
            input_name=input_name_lanesod,
            input_shape=input_shape_lanesod,
            image=debug_image,
        )



        output_image = cv.resize(
            output_image,
            dsize=(
                debug_image.shape[1],
                debug_image.shape[0],
            )
        )

        # Inference elapsed time
        cv.putText(
            output_image,
            f"Elapsed Time : {elapsed_time * 1000:.1f} ms",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )
        cv.putText(
            output_image,
            f"Elapsed Time : {elapsed_time * 1000:.1f} ms",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            1,
            cv.LINE_AA,
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        combined_img = np.vstack([output_nodehaze_image, output_dehaze_image])
        cv.imshow(WINDOW_NAME, combined_img)
        video_writer.write(combined_img)

    if video_writer is not None:
        video_writer.release()
    if cap is not None:
        cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()