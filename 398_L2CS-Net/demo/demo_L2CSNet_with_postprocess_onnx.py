#!/usr/bin/env python

import cv2
import copy
import time
import psutil
import argparse
import numpy as np
import onnxruntime
from typing import Optional, List, Tuple


class RetinaFaceONNX(object):
    def __init__(
        self,
        model_path: Optional[str] = 'retinaface_mbn025_with_postprocess_480x640_max1000_th0.70.onnx',
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
    ) -> np.ndarray:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
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

class L2CSNetONNX(object):
    def __init__(
        self,
        model_path: Optional[str] = 'l2cs_net_Nx3x448x448.onnx',
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
        """L2CSNETONNX

        Parameters
        ----------
        model_path: Optional[str]
            ONNX file path for L2CS-Net

        providers: Optional[List]
            Name of onnx execution providers
        """
        # Model loading
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        session_option.intra_op_num_threads = psutil.cpu_count(logical=True) - 1
        self.onnx_session = \
            onnxruntime.InferenceSession(
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

        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

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
        gaze_yaw_pitch: np.ndarray
            Gaze yaw pitch: [facecount, [Yaw, Pitch]]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_images = \
            self.__preprocess(
                temp_image,
            )

        # Inference
        gaze_yaw_pitch = \
            self.onnx_session.run(
                self.output_names,
                {input_name: resized_images for input_name in self.input_names},
            )[0]
        return gaze_yaw_pitch


    def __preprocess(
        self,
        images: List[np.ndarray],
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

        resized_images = []

        for image in images:
            # Normalization + BGR->RGB
            resized_image = cv2.resize(
                image,
                (
                    int(self.input_shapes[0][3]),
                    int(self.input_shapes[0][2]),
                )
            )
            resized_image = np.divide(resized_image, 255.0)
            resized_image = (resized_image - self.mean) / self.std
            resized_image = resized_image[..., ::-1]
            resized_image = resized_image.transpose(swap)
            resized_image = \
                np.ascontiguousarray(
                    resized_image,
                    dtype=np.float32,
                )
            resized_images.append(resized_image)
        return np.asarray(resized_images).astype(np.float32)

def draw_gaze(
    a,
    b,
    c,
    d,
    image_in,
    yaw_pitch,
    thickness=2,
    color=(255, 255, 0),
):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w/2
    pos = (int(a+c / 2.0), int(b+d / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(yaw_pitch[0]) * np.cos(yaw_pitch[1])
    dy = -length * np.sin(yaw_pitch[1])
    cv2.arrowedLine(
        image_out,
        tuple(np.round(pos).astype(np.int32)),
        tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)),
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.18,
    )
    return image_out


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
        '-dm',
        '--detector_model',
        type=str,
        default='retinaface_mbn025_with_postprocess_480x640_max1000_th0.70.onnx',
    )
    parser.add_argument(
        '-pm',
        '--predictor_model',
        type=str,
        default='l2cs_net_Nx3x448x448.onnx',
    )
    parser.add_argument(
        '-p',
        '--provider',
        type=str,
        default='cuda',
        choices=['cpu','cuda','tensorrt'],
    )
    args = parser.parse_args()

    # Initialize video capture
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
        model_path=args.detector_model,
        providers=providers,
    )

    # Predictor Model
    predictor = L2CSNetONNX(
        model_path=args.predictor_model,
        providers=providers,
    )

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        debug_image = copy.deepcopy(frame)
        input_image = copy.deepcopy(frame)

        image_height = debug_image.shape[0]
        image_width = debug_image.shape[1]

        # Face Detection
        batchno_classid_score_x1y1x2y2_landms = detector(input_image)

        imgs = []
        for batchno_classid_score_x1y1x2y2_landm in batchno_classid_score_x1y1x2y2_landms:
            x_min = max(int(batchno_classid_score_x1y1x2y2_landm[3]), 0)
            y_min = max(int(batchno_classid_score_x1y1x2y2_landm[4]), 0)
            x_max = min(int(batchno_classid_score_x1y1x2y2_landm[5]), image_width)
            y_max = min(int(batchno_classid_score_x1y1x2y2_landm[6]), image_height)
            imgs.append(input_image[y_min:y_max, x_min:x_max])

        # Inference execution
        #
        # gaze_yaw_pitchs.shape
        #   float32 [1, 2]
        gaze_yaw_pitches = []
        if len(imgs) > 0:
            gaze_yaw_pitches = predictor(imgs)

        elapsed_time = time.time() - start_time

        # Draw
        face_boxes = batchno_classid_score_x1y1x2y2_landms[..., 3:]
        for bbox, gaze_yaw_pitch in zip(face_boxes, gaze_yaw_pitches):
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])
            cv2.rectangle(
                debug_image,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2,
            )
            draw_gaze(
                a=x1,
                b=y1,
                c=x2-x1,
                d=y2-y1,
                image_in=debug_image,
                yaw_pitch=(gaze_yaw_pitch[0], gaze_yaw_pitch[1]),
                color=(0, 0, 255)
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
        cv2.imshow('L2CS-Net', debug_image)
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