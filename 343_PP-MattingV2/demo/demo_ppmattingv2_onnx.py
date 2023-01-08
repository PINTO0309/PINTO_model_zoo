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
    # ONNX Infomation
    input_width = input_size[3]
    input_height = input_size[2]
    # Pre process:Resize, BGR->RGB, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = input_image[..., ::-1]
    input_image = input_image.transpose(2, 0, 1)[np.newaxis, ...]
    input_image = input_image / 255.0
    input_image = input_image.astype('float32')
    # Inference
    segmentation_map = \
        onnx_session.run(
            None,
            {
                input_name: input_image,
            },
        )[0]
    segmentation_map = segmentation_map[0]
    segmentation_map = segmentation_map.transpose(1, 2, 0)
    return segmentation_map


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
        default='ppmattingv2_stdc1_human_480x640.onnx',
    )
    parser.add_argument(
        '-p',
        '--provider',
        type=str,
        default='cuda',
        choices=['cpu','cuda','tensorrt'],
    )
    parser.add_argument(
        '-s',
        '--score_threshold',
        type=float,
        default=0.65,
    )
    args = parser.parse_args()
    device: int = args.device
    movie: str = args.movie
    model: str = args.model
    provider: str = args.provider
    score_threshold: float = args.score_threshold
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
        frameSize=(cap_width, cap_height),
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

        # Inference execution
        segmentation_map = run_inference(
            onnx_session=onnx_session,
            input_name=input_name,
            input_size=input_size,
            image=frame,
        )

        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            score_threshold,
            segmentation_map,
        )

        video_writer.write(debug_image)
        cv.imshow(f'PP-MattingV2 ONNX', debug_image)
        key = cv.waitKey(1) if movie is None or movie[-4:].lower() == '.mp4' else cv.waitKey(0)
        if key == 27:  # ESC
            break

    if video_writer:
        video_writer.release()
    if cap:
        cap.release()
    cv.destroyAllWindows()


def draw_debug(image, elapsed_time, score, segmentation_map):
    image_width, image_height = image.shape[1], image.shape[0]

    # Match the size
    debug_image = copy.deepcopy(image)
    segmentation_map = cv.resize(
        segmentation_map,
        dsize=(image_width, image_height),
        interpolation=cv.INTER_LINEAR,
    )
    segmentation_map = segmentation_map[..., np.newaxis]

    # color list
    color_image_list = []
    # ID 0:BackGround
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = (0, 0, 0)
    color_image_list.append(bg_image)
    # ID 1:Human
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = (0, 255, 0)
    color_image_list.append(bg_image)

    # Overlay segmentation map
    masks = segmentation_map.transpose(2, 0, 1)
    for index, mask in enumerate(masks):
        # Threshold check by score
        mask = np.where(mask > score, 0, 1)

        # Overlay
        mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
        mask_image = np.where(mask, debug_image, color_image_list[index+1])
        debug_image = cv.addWeighted(debug_image, 0.5, mask_image, 0.5, 1.0)

    # Inference elapsed time
    cv.putText(
        debug_image,
        "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
        (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        cv.LINE_AA,
    )

    return debug_image


if __name__ == '__main__':
    main()
