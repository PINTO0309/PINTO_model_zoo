#!/usr/bin/env python

import copy
import time
import argparse
from typing import Optional, List, Tuple

import cv2 as cv
import psutil
import numpy as np
import onnxruntime


class EarSegmentationONNX(object):
    def __init__(
        self,
        model_path: Optional[str] = 'ear_segmentation_1x3x480x640.onnx',
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
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

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
        batchno_classid_score_x1y1x2y2_landms: np.ndarray
            [N, [batchno, classid, score, x1, y1, x2, y2, landms0, ..., landms9]]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = \
            self.__preprocess(
                temp_image,
            )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=np.float32)
        segmentation_map = \
            self.onnx_session.run(
                self.output_names,
                {input_name: inferece_image for input_name in self.input_names},
            )[0]
        segmentation_map = segmentation_map[0]
        segmentation_map = segmentation_map.transpose(1, 2, 0)
        return segmentation_map

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
        resized_image = cv.resize(
            image,
            (
                int(self.input_shapes[0][3]),
                int(self.input_shapes[0][2]),
            )
        )
        resized_image = resized_image[..., ::-1]
        resized_image = resized_image / 255.0
        resized_image = (resized_image - self.mean) / self.std
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
        default='ear_segmentation_1x3x480x640.onnx',
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
    score_threshold = args.score_threshold

    cap = cv.VideoCapture(cap_device)
    cap_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    cap_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    cap_fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter.fourcc('m','p','4','v')
    video_writer = cv.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(cap_width, cap_height),
    )

    segmentor = EarSegmentationONNX(
        model_path=args.model,
        providers=providers,
    )

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break

        debug_image = copy.deepcopy(frame)

        start_time = time.time()
        segmentation_map = segmentor(debug_image)
        elapsed_time = time.time() - start_time

        # Draw
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            score_threshold,
            segmentation_map,
        )

        video_writer.write(debug_image)
        cv.imshow(f'Ear Segmentation ({args.provider})', debug_image)
        key = cv.waitKey(1) \
            if args.movie is None or args.movie[-4:] == '.mp4' else cv.waitKey(0)
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
