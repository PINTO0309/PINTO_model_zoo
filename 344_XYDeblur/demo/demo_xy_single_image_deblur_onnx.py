#!/usr/bin/env python

import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime
from typing import Tuple, List


def run_inference(
    onnx_session: onnxruntime.InferenceSession,
    input_name: str,
    input_size: List[int],
    image: np.ndarray,
) -> Tuple[np.ndarray,np.ndarray]:
    h = image.shape[0]
    w = image.shape[1]
    # ONNX Infomation
    input_width = input_size[3]
    input_height = input_size[2]
    # Pre process:Resize, NHWC->NCHW, BGR->RGB, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = input_image[..., ::-1]
    input_image = input_image.transpose(2, 0, 1)[np.newaxis, ...]
    input_image = input_image / 255.0
    input_image = input_image.astype('float32')
    # Inference
    deblured_image = \
        onnx_session.run(
            None,
            {input_name: input_image},
        )[0]
    # Post Process, NCHW->NHWC, RGB->BGR, uint8 cast
    deblured_image = cv.resize(
        src=deblured_image[0].transpose(1,2,0)[..., ::-1].astype(np.uint8),
        dsize=[w,h],
    )
    return deblured_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--device',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-mov',
        '--movie',
        type=str,
        default=None,
    )
    parser.add_argument(
        '-mod',
        '--model',
        type=str,
        default='xy_single_image_deblur_480x640.onnx',
    )
    parser.add_argument(
        '-p',
        '--provider',
        type=str,
        default='cpu',
        choices=['cpu','cuda','tensorrt'],
    )
    args = parser.parse_args()
    device: int = args.device
    movie: str = args.movie
    model: str = args.model
    provider: str = args.provider

    # Initialize video capture
    cap_device = device
    if movie is not None:
        cap_device = movie
    cap = cv.VideoCapture(cap_device)
    cap_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    cap_fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc('m','p','4','v')
    video_writer = cv.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(cap_width, cap_height*2),
    )

    # Load model
    providers = []
    if provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif provider == 'cuda':
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif provider == 'tensorrt':
        providers = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]

    onnx_session = onnxruntime.InferenceSession(
        path_or_bytes=model,
        providers=providers,
    )
    input_name: str = onnx_session.get_inputs()[0].name
    input_size: List[int] = onnx_session.get_inputs()[0].shape

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)
        start_time = time.time()

        # Deblur
        deblured_image = run_inference(
            onnx_session=onnx_session,
            input_name=input_name,
            input_size=input_size,
            image=debug_image,
        )

        elapsed_time = time.time() - start_time

        # Inference elapsed time
        cv.putText(
            deblured_image,
            f'Elapsed Time: {elapsed_time * 1000:.1f} ms',
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            1,
            cv.LINE_AA,
        )

        view_image = np.vstack([debug_image, deblured_image])
        video_writer.write(view_image)
        cv.imshow(f'XYDeblur ({model} {provider}) ONNX', view_image)
        key = cv.waitKey(1) if movie is None or movie[-4:] == '.mp4' else cv.waitKey(0)
        if key == 27:  # ESC
            break

    if video_writer:
        video_writer.release()
    if cap:
        cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
