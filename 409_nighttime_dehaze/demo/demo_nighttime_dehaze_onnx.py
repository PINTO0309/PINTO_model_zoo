#!/usr/bin/env python

import copy
import time
import argparse
from typing import Optional, List, Tuple

import cv2
import psutil
import numpy as np
import onnxruntime


class NighttimeDehazeONNX(object):
    def __init__(
        self,
        model_path: Optional[str] = 'nighttime_dehaze_realnight_1x3x512x512.onnx',
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
        """NighttimeDehazeONNX

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
        processed_image: np.ndarray
            Dehazed image
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = \
            self.__preprocess(
                temp_image,
            )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=np.float32)
        result = \
            self.onnx_session.run(
                self.output_names,
                {input_name: inferece_image for input_name in self.input_names},
            )[0]
        processed_image = self.__postprocess(result)
        return processed_image

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
        self.image_height = image.shape[0]
        self.input_height = self.image_height
        if not isinstance(self.input_shapes[0][2], str):
            self.input_height = self.input_shapes[0][2]
        self.image_width = image.shape[1]
        self.input_width = self.image_width
        if not isinstance(self.input_shapes[0][3], str):
            self.input_width = self.input_shapes[0][3]

        # Normalization + BGR->RGB
        resized_image = cv2.resize(
            image,
            (
                int(self.input_width),
                int(self.input_height),
            )
        )
        resized_image = resized_image[..., ::-1]
        resized_image = (resized_image / 255.0 - 0.5) / 0.5
        resized_image = resized_image.transpose(swap)
        resized_image = \
            np.ascontiguousarray(
                resized_image,
                dtype=np.float32,
            )
        return resized_image


    def __postprocess(
        self,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (1, 2, 0),
    ) -> np.ndarray:
        """__postprocess

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
        image = image[0]
        image = image.transpose(swap)
        resized_image = cv2.resize(
            image,
            (
                int(self.image_width),
                int(self.image_height),
            )
        )
        return resized_image.astype(np.uint8)


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
        default='nighttime_dehaze_realnight_1x3x512x512.onnx',
    )
    parser.add_argument(
        '-p',
        '--provider',
        type=str,
        default='cuda',
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
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter.fourcc('m','p','4','v')
    video_writer = cv2.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(cap_width, cap_height),
    )

    dehazer = NighttimeDehazeONNX(
        model_path=args.model,
        providers=providers,
    )

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break

        original_image = copy.deepcopy(frame)
        debug_image = copy.deepcopy(frame)

        start_time = time.time()
        debug_image = dehazer(debug_image)
        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            original_image,
            debug_image,
        )

        video_writer.write(debug_image)
        cv2.imshow(f'Nighttime Dehaze ({args.provider})', debug_image)
        key = cv2.waitKey(1) \
            if args.movie is None or args.movie[-4:] == '.mp4' else cv2.waitKey(0)
        if key == 27:  # ESC
            if args.movie[-4:] == '.jpg' or args.movie[-4:] == '.png':
                cv2.imwrite('test.png', debug_image)

    if video_writer:
        video_writer.release()
    if cap:
        cap.release()
    cv2.destroyAllWindows()


def draw_debug(
    original_image,
    debug_image,
):
    # Match the size
    debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
    combined_img = np.vstack([original_image, debug_image])

    return combined_img


if __name__ == '__main__':
    main()
