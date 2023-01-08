#!/usr/bin/env python

import copy
import time
import argparse
import collections

import cv2 as cv
import numpy as np
import onnxruntime
from typing import Tuple, List, Any


def run_inference(
    onnx_session: onnxruntime.InferenceSession,
    input_name_image: str,
    input_size_image: List[int],
    input_name_prev_keypoints: str,
    input_name_prev_descriptors: str,
    prev_keypoints: np.ndarray,
    prev_descriptors: np.ndarray,
    image: np.ndarray,
) -> Tuple[np.ndarray,np.ndarray]:
    # ONNX Infomation
    input_width = input_size_image[3]
    input_height = input_size_image[2]
    # Pre process:Resize, BGR->RGB, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = input_image[..., ::-1]
    input_image = input_image.transpose(2, 0, 1)[np.newaxis, ...]
    input_image = input_image / 255.0
    input_image = input_image.astype('float32')
    # Inference
    matched_keypoints1_xy, matched_keypoints2_xy, keypoints, descriptors = \
        onnx_session.run(
            None,
            {
                input_name_image: input_image,
                input_name_prev_keypoints: prev_keypoints,
                input_name_prev_descriptors: prev_descriptors,
            },
        )
    return matched_keypoints1_xy, matched_keypoints2_xy, keypoints, descriptors


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
        default='alike_t_opset16_480x640_post.onnx',
    )
    parser.add_argument(
        '-p',
        '--provider',
        type=str,
        default='cpu',
        choices=['cpu','cuda','tensorrt'],
    )
    parser.add_argument(
        '-s',
        '--skip_frame_count',
        type=int,
        default=0,
        help='skip_frame_count+1 value of whether the feature point is compared to the previous frame.'
    )
    args = parser.parse_args()
    device: int = args.device
    movie: str = args.movie
    model: str = args.model
    provider: str = args.provider
    skip_frame_count: int = args.skip_frame_count

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
    input_name_image: str = onnx_session.get_inputs()[0].name
    input_size_image: List[int] = onnx_session.get_inputs()[0].shape
    input_name_prev_keypoints: str = onnx_session.get_inputs()[1].name
    input_name_prev_descriptors: str = onnx_session.get_inputs()[2].name
    input_size_prev_descriptors: List[Any] = onnx_session.get_inputs()[2].shape
    in_process_flag = True

    # First time capture read
    ret, frame = cap.read()
    if not ret:
        in_process_flag = False

    # First time inference
    prev_keypoints = None
    prev_descriptors = None
    buffer_count = skip_frame_count + 1
    keypoints_descriptors_buffer = collections.deque([], buffer_count)
    for _ in range(buffer_count):
        _, _, prev_keypoints, prev_descriptors = \
            run_inference(
                onnx_session=onnx_session,
                input_name_image=input_name_image,
                input_size_image=input_size_image,
                input_name_prev_keypoints=input_name_prev_keypoints,
                input_name_prev_descriptors=input_name_prev_descriptors,
                prev_keypoints=np.zeros([5000, 2], dtype=np.float32),
                prev_descriptors=np.zeros([5000, input_size_prev_descriptors[-1]], dtype=np.float32),
                image=frame,
            )
        keypoints_descriptors_buffer.append(
            [prev_keypoints, prev_descriptors]
        )

    while in_process_flag:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)
        start_time = time.time()

        # Keypoint detection
        prev_keypoints_prev_descriptors: List = keypoints_descriptors_buffer.popleft()
        matched_keypoints1_xy, matched_keypoints2_xy, keypoints, descriptors = \
            run_inference(
                onnx_session=onnx_session,
                input_name_image=input_name_image,
                input_size_image=input_size_image,
                input_name_prev_keypoints=input_name_prev_keypoints,
                input_name_prev_descriptors=input_name_prev_descriptors,
                prev_keypoints=prev_keypoints_prev_descriptors[0],
                prev_descriptors=prev_keypoints_prev_descriptors[1],
                image=frame,
            )
        keypoints_descriptors_buffer.append(
            [keypoints, descriptors]
        )

        elapsed_time = time.time() - start_time

        _ = [
            (
                cv.line(
                    debug_image,
                    (int(round(matched_keypoint1[0])), int(round(matched_keypoint1[1]))),
                    (int(round(matched_keypoint2[0])), int(round(matched_keypoint2[1]))),
                    (0, 255, 0),
                    lineType=16
                ), \
                cv.circle(
                    debug_image,
                    (int(round(matched_keypoint2[0])), int(round(matched_keypoint2[1]))),
                    1,
                    (0, 0, 255),
                    -1,
                    lineType=16
                )
            ) \
            for matched_keypoint1, matched_keypoint2 in zip(matched_keypoints1_xy, matched_keypoints2_xy) \
                if int(round(matched_keypoint1[0])) > 0 \
                    and int(round(matched_keypoint1[1])) > 0 \
                    and int(round(matched_keypoint2[0])) > 0 \
                    and int(round(matched_keypoint2[1])) > 0
        ]

        # Inference elapsed time
        cv.putText(
            debug_image,
            f'Elapsed Time: {elapsed_time * 1000:.1f} ms',
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            1,
            cv.LINE_AA,
        )

        video_writer.write(debug_image)
        cv.imshow(f'ALIKE ({model} {provider}) ONNX', debug_image)
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
