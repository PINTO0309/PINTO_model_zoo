#!/usr/bin/env python

"""
This demo code is for evaluation of an end-to-end model with
post-processing merged. Automatic determination of the runtime
that should be executed based on the file extension.

runtime: https://github.com/microsoft/onnxruntime
runtime: https://github.com/PINTO0309/TensorflowLite-bin
"""
from __future__ import annotations
import os
import sys
import copy
import cv2
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import Tuple, Optional, List, Dict
import importlib.util
from abc import ABC, abstractmethod

class Color(Enum):
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERSE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

    def __str__(self):
        return self.value

    def __call__(self, s):
        return str(self) + str(s) + str(Color.RESET)

@dataclass(frozen=False)
class Box():
    classid: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int

class AbstractModel(ABC):
    """AbstractModel
    Base class of the model.
    """
    _runtime: str = 'onnx'
    _model_path: str = ''
    _class_score_th: float = 0.35
    _input_shapes: List[List[int]] = []
    _input_names: List[str] = []
    _output_shapes: List[List[int]] = []
    _output_names: List[str] = []

    # onnx/tflite
    _interpreter = None
    _inference_model = None
    _providers = None
    _swap = (2, 0, 1)
    _h_index = 2
    _w_index = 3

    # onnx
    _onnx_dtypes_to_np_dtypes = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
    }

    # tflite
    _input_details = None
    _output_details = None

    @abstractmethod
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = '',
        class_score_th: Optional[float] = 0.35,
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
        self._runtime = runtime
        self._model_path = model_path
        self._class_score_th = class_score_th
        self._providers = providers

        # Model loading
        if self._runtime == 'onnx':
            import onnxruntime # type: ignore
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self._interpreter = \
                onnxruntime.InferenceSession(
                    model_path,
                    sess_options=session_option,
                    providers=providers,
                )
            self._providers = self._interpreter.get_providers()
            self._input_shapes = [
                input.shape for input in self._interpreter.get_inputs()
            ]
            self._input_names = [
                input.name for input in self._interpreter.get_inputs()
            ]
            self._input_dtypes = [
                self._onnx_dtypes_to_np_dtypes[input.type] for input in self._interpreter.get_inputs()
            ]
            self._output_shapes = [
                output.shape for output in self._interpreter.get_outputs()
            ]
            self._output_names = [
                output.name for output in self._interpreter.get_outputs()
            ]
            self._model = self._interpreter.run
            self._swap = (2, 0, 1)
            self._h_index = 2
            self._w_index = 3

        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            if self._runtime == 'tflite_runtime':
                from tflite_runtime.interpreter import Interpreter # type: ignore
                self._interpreter = Interpreter(model_path=model_path)
            elif self._runtime == 'tensorflow':
                import tensorflow as tf # type: ignore
                self._interpreter = tf.lite.Interpreter(model_path=model_path)
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._input_shapes = [
                input.get('shape', None) for input in self._input_details
            ]
            self._input_names = [
                input.get('name', None) for input in self._input_details
            ]
            self._input_dtypes = [
                input.get('dtype', None) for input in self._input_details
            ]
            self._output_shapes = [
                output.get('shape', None) for output in self._output_details
            ]
            self._output_names = [
                output.get('name', None) for output in self._output_details
            ]
            self._model = self._interpreter.get_signature_runner()
            self._swap = (0, 1, 2)
            self._h_index = 1
            self._w_index = 2

    @abstractmethod
    def __call__(
        self,
        *,
        input_datas: List[np.ndarray],
    ) -> List[np.ndarray]:
        datas = {
            f'{input_name}': input_data \
                for input_name, input_data in zip(self._input_names, input_datas)
        }
        if self._runtime == 'onnx':
            outputs = [
                output for output in \
                    self._model(
                        output_names=self._output_names,
                        input_feed=datas,
                    )
            ]
            return outputs
        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            outputs = [
                output for output in \
                    self._model(
                        **datas
                    ).values()
            ]
            return outputs

    @abstractmethod
    def _preprocess(
        self,
        *,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _postprocess(
        self,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        pass

class DepthAnything(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'depth_anything_vits14_1x3x480x640.onnx',
        providers: Optional[List] = None,
    ):
        """

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for YOLOX. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for YOLOX

        class_score_th: Optional[float]
            Score threshold. Default: 0.35

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            providers=providers,
        )
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

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
        depth_image: np.ndarray
            [N, 1, H, W]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        inferece_image = \
            self._preprocess(
                temp_image,
            )

        # Inference
        outputs = super().__call__(input_datas=[inferece_image])
        depth_image = outputs[0]

        # PostProcess
        depth_image = \
            self._postprocess(
                image=depth_image,
            )

        return depth_image

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """_preprocess

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
        # Resize + Transpose
        resized_image: np.ndarray = \
            cv2.resize(
                image,
                (
                    int(self._input_shapes[0][self._w_index]),
                    int(self._input_shapes[0][self._h_index]),
                )
            )
        resized_image = resized_image[..., ::-1]
        resized_image = resized_image.transpose(self._swap)
        resized_image = (resized_image / 255.0 - self.mean) / self.std
        resized_image = resized_image.astype(self._input_dtypes[0])
        return resized_image

    def _postprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """_postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        Returns
        -------
        depth_image: np.ndarray
            [N, 1, H, W]
        """
        image = image[0]
        # (2, 0, 1)
        reverse_swap = tuple([self._swap.index(i) for i in range(len(self._swap))])
        image = image.transpose(reverse_swap)
        depth_image: np.ndarray = cv2.resize(image, (self._input_shapes[0][self._w_index], self._input_shapes[0][self._h_index]))
        return depth_image

def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_package_installed(package_name: str):
    """Checks if the specified package is installed.

    Parameters
    ----------
    package_name: str
        Name of the package to be checked.

    Returns
    -------
    result: bool
        True if the package is installed, false otherwise.
    """
    return importlib.util.find_spec(package_name) is not None

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='depth_anything_vits14_1x3x480x640.onnx',
        help='ONNX/TFLite file path for DepthAnything.',
    )
    parser.add_argument(
        '-v',
        '--video',
        type=str,
        default="0",
        help='Video file path or camera index.',
    )
    parser.add_argument(
        '-ep',
        '--execution_provider',
        type=str,
        choices=['cpu', 'cuda', 'tensorrt'],
        default='cuda',
        help='Execution provider for ONNXRuntime.',
    )
    parser.add_argument(
        '-dvw',
        '--disable_video_writer',
        action='store_true',
        help=\
            'Disable video writer. '+
            'Eliminates the file I/O load associated with automatic recording to MP4. '+
            'Devices that use a MicroSD card or similar for main storage can speed up overall processing.',
    )
    args = parser.parse_args()

    # runtime check
    model_file: str = args.model
    model_ext: str = os.path.splitext(model_file)[1][1:].lower()
    runtime: str = None
    if model_ext == 'onnx':
        if not is_package_installed('onnxruntime'):
            print(Color.RED('ERROR: onnxruntime is not installed. pip install onnxruntime or pip install onnxruntime-gpu'))
            sys.exit(0)
        runtime = 'onnx'
    elif model_ext == 'tflite':
        if is_package_installed('tflite_runtime'):
            runtime = 'tflite_runtime'
        elif is_package_installed('tensorflow'):
            runtime = 'tensorflow'
        else:
            print(Color.RED('ERROR: tflite_runtime or tensorflow is not installed.'))
            print(Color.RED('ERROR: https://github.com/PINTO0309/TensorflowLite-bin'))
            print(Color.RED('ERROR: https://github.com/tensorflow/tensorflow'))
            sys.exit(0)
    video: str = args.video
    execution_provider: str = args.execution_provider
    providers: List[Tuple[str, Dict] | str] = None
    if execution_provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'cuda':
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'tensorrt':
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

    # Model initialization
    model = DepthAnything(
        runtime=runtime,
        model_path=model_file,
        providers=providers,
    )

    cap = cv2.VideoCapture(
        int(video) if is_parsable_to_int(video) else video
    )
    disable_video_writer: bool = args.disable_video_writer
    video_writer = None
    if not disable_video_writer:
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            filename='output.mp4',
            fourcc=fourcc,
            fps=cap_fps,
            frameSize=(w, h),
        )

    while cap.isOpened():
        res, image = cap.read()
        if not res:
            break

        debug_image = copy.deepcopy(image)
        start_time = time.perf_counter()
        depth_image: np.ndarray = model(debug_image)
        elapsed_time = time.perf_counter() - start_time
        depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255.0
        depth_image = depth_image.astype(np.uint8)
        depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_INFERNO)
        cv2.putText(depth_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(depth_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        key = cv2.waitKey(1)
        if key == 27: # ESC
            break

        cv2.imshow("test", depth_image)
        if video_writer is not None:
            video_writer.write(depth_image)

    if video_writer is not None:
        video_writer.release()

    if cap is not None:
        cap.release()

if __name__ == "__main__":
    main()
