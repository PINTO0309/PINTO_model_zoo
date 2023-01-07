#!/usr/bin/env python

import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime
from typing import Tuple, List


class SimpleTracker(object):
    def __init__(self):
        self.prev_keypoints = None
        self.prev_descriptors = None

    def update(
        self,
        keypoints: np.ndarray,
        descriptors: np.ndarray,
    ) -> Tuple[int,np.ndarray,np.ndarray]:
        matched_count = 0
        matched_keypoints1 = np.asarray([])
        matched_keypoints2 = np.asarray([])
        if self.prev_keypoints is None:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
        else:
            matches = self.mnn_matcher(self.prev_descriptors, descriptors)
            matched_keypoints1 = self.prev_keypoints[matches[:, 0]]
            matched_keypoints2 = keypoints[matches[:, 1]]
            matched_count = len(matches)
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
        return matched_count, matched_keypoints1, matched_keypoints2

    def mnn_matcher(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ) -> np.ndarray:
        sim = desc1 @ desc2.transpose()
        sim[sim < 0.9] = 0
        nn12 = np.argmax(sim, axis=1)
        nn21 = np.argmax(sim, axis=0)
        ids1 = np.arange(0, sim.shape[0])
        mask = (ids1 == nn21[nn12])
        matches = np.stack([ids1[mask], nn12[mask]])
        return matches.transpose()


def run_inference(
    onnx_session: onnxruntime.InferenceSession,
    input_name: str,
    input_size: List[int],
    image: np.ndarray,
    score_th: float=0.2,
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
    keypoints, descriptors, scores, scores_map = \
        onnx_session.run(
            None,
            {input_name: input_image},
        )
    # Post process
    indicies = scores[:, 0] > score_th
    keypoints = keypoints[indicies, :]
    descriptors = descriptors[indicies, :]
    return keypoints, descriptors


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
        default='alike_t_opset16_480x640.onnx',
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
        '--score_th',
        type=float,
        default=0.2,
    )
    args = parser.parse_args()
    device: int = args.device
    movie: str = args.movie
    model: str = args.model
    provider: str = args.provider
    score_th: float = args.score_th

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

    tracker = SimpleTracker()

    while True:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Keypoint detection
        keypoints, descriptors = run_inference(
            onnx_session=onnx_session,
            input_name=input_name,
            input_size=input_size,
            image=frame,
            score_th=score_th,
        )

        # Tracker update
        matched_count, matched_keypoints1, matched_keypoints2 = \
            tracker.update(
                keypoints=keypoints,
                descriptors=descriptors,
            )

        if tracker.prev_keypoints is None:
            for keypoint in keypoints:
                p1 = (int(round(keypoint[0])), int(round(keypoint[1])))
                cv.circle(debug_image, p1, 1, (0, 0, 255), -1, lineType=16)
        else:
            for matched_keypoint1, matched_keypoint2 in zip(matched_keypoints1, matched_keypoints2):
                p1 = (int(round(matched_keypoint1[0])), int(round(matched_keypoint1[1])))
                p2 = (int(round(matched_keypoint2[0])), int(round(matched_keypoint2[1])))
                cv.line(debug_image, p1, p2, (0, 255, 0), lineType=16)
                cv.circle(debug_image, p2, 1, (0, 0, 255), -1, lineType=16)

        elapsed_time = time.time() - start_time

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