#!/usr/bin/env python

import copy
import time
import argparse
from typing import Optional, List, Tuple

import cv2
import psutil
import numpy as np
import onnxruntime


class RetinaFaceONNX(object):
    def __init__(
        self,
        model_path: Optional[str] = 'retinaface_mbn025_with_postprocess_480x640_max1000.onnx',
        providers: Optional[List] = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        """RetinaFaceONNX

        Parameters
        ----------
        model_path: Optional[str]
            ONNX file path

        providers: Optional[List]
            Name of onnx execution providers
        """
        # Model loading
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        session_option.intra_op_num_threads = psutil.cpu_count(logical=True) - 1
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=providers,
        )
        self.providers = self.onnx_session.get_providers()

        self.input_shapes = [
            input.shape for input in self.onnx_session.get_inputs()
        ]
        self.input_names = [
            input.name for input in self.onnx_session.get_inputs()
        ]
        self.output_names = [
            output.name for output in self.onnx_session.get_outputs()
        ]
        self.mean = np.asarray([104, 117, 123], dtype=np.float32)

    def __call__(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        face_boxes: np.ndarray
            Predicted face boxes: [facecount, y1, x1, y2, x2]

        face_scores: np.ndarray
            Predicted face box scores: [facecount, score]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = \
            self.__preprocess(
                temp_image,
            )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=np.float32)
        batchno_classid_score_x1y1x2y2_landms = \
            self.onnx_session.run(
                self.output_names,
                {input_name: inferece_image for input_name in self.input_names},
            )[0]

        return batchno_classid_score_x1y1x2y2_landms

    def __preprocess(
        self,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        """__preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        # Normalization + BGR->RGB
        resized_image = cv2.resize(
            image,
            (
                int(self.input_shapes[0][3]),
                int(self.input_shapes[0][2]),
            )
        )
        resized_image = resized_image[..., ::-1]
        resized_image = (resized_image - self.mean)
        resized_image = resized_image.transpose(swap)
        resized_image = \
            np.ascontiguousarray(
                resized_image,
                dtype=np.float32,
            )
        return resized_image


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
        '-m',
        '--model',
        type=str,
        default='retinaface_mbn025_with_postprocess_480x640_max1000.onnx',
    )
    parser.add_argument(
        '-p',
        '--provider',
        type=str,
        default='cpu',
        choices=['cpu','cuda','tensorrt'],
    )
    args = parser.parse_args()

    cap_device: int = args.device
    if args.movie is not None:
        cap_device = args.movie

    providers = None
    if args.provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif args.provider == 'cuda':
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif args.provider == 'tensorrt':
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

    cap = cv2.VideoCapture(cap_device)
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter.fourcc('m','p','4','v')
    video_writer = cv2.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(cap_width, cap_height),
    )

    # Detector Model: RetinaFace
    detector = RetinaFaceONNX(
        model_path=args.model,
        providers=providers,
    )

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break

        debug_image = copy.deepcopy(frame)

        image_height = debug_image.shape[0]
        image_width = debug_image.shape[1]

        # Face Detection
        start_time = time.time()
        batchno_classid_score_x1y1x2y2_landms = detector(debug_image)
        elapsed_time = time.time() - start_time

        for batchno_classid_score_x1y1x2y2_landm in batchno_classid_score_x1y1x2y2_landms:
            x_min = max(int(batchno_classid_score_x1y1x2y2_landm[3]), 0)
            y_min = max(int(batchno_classid_score_x1y1x2y2_landm[4]), 0)
            x_max = min(int(batchno_classid_score_x1y1x2y2_landm[5]), image_width)
            y_max = min(int(batchno_classid_score_x1y1x2y2_landm[6]), image_height)
            # Face
            cv2.rectangle(
                debug_image,
                (x_min, y_min),
                (x_max, y_max),
                (255, 255, 0),
                2,
            )
            # Landms
            cv2.circle(
                debug_image,
                (
                    int(batchno_classid_score_x1y1x2y2_landm[7]),
                    int(batchno_classid_score_x1y1x2y2_landm[8])
                ),
                1,
                (0, 0, 255),
                4,
            )
            cv2.circle(
                debug_image,
                (
                    int(batchno_classid_score_x1y1x2y2_landm[9]),
                    int(batchno_classid_score_x1y1x2y2_landm[10])
                ),
                1,
                (0, 255, 255),
                4,
            )
            cv2.circle(
                debug_image,
                (
                    int(batchno_classid_score_x1y1x2y2_landm[11]),
                    int(batchno_classid_score_x1y1x2y2_landm[12])
                ),
                1,
                (255, 0, 255),
                4,
            )
            cv2.circle(
                debug_image,
                (
                    int(batchno_classid_score_x1y1x2y2_landm[13]),
                    int(batchno_classid_score_x1y1x2y2_landm[14])
                ),
                1,
                (0, 255, 0),
                4,
            )
            cv2.circle(
                debug_image,
                (
                    int(batchno_classid_score_x1y1x2y2_landm[15]),
                    int(batchno_classid_score_x1y1x2y2_landm[16])
                ),
                1,
                (255, 0, 0),
                4,
            )

        # Inference elapsed time
        cv2.putText(
            debug_image,
            f'Elapsed Time: {elapsed_time * 1000:.1f} ms',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        video_writer.write(debug_image)
        cv2.imshow('RetinaFace MobileNetv2', debug_image)
        key = cv2.waitKey(1) \
            if args.movie is None or args.movie[-4:] == '.mp4' else cv2.waitKey(0)
        if key == 27:  # ESC
            break

    if video_writer:
        video_writer.release()
    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
