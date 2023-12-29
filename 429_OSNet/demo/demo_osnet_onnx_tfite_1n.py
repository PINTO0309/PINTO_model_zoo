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

class AbstractModel(ABC):
    """AbstractModel
    Base class of the model.
    """
    _runtime: str = 'onnx'
    _model_path: str = ''
    _input_shapes: List[List[int]] = []
    _input_names: List[str] = []
    _output_shapes: List[List[int]] = []
    _output_names: List[str] = []

    _mean: np.ndarray = np.array([0.000, 0.000, 0.000], dtype=np.float32)
    _std: np.ndarray = np.array([1.000, 1.000, 1.000], dtype=np.float32)

    # onnx/tflite
    _interpreter = None
    _inference_model = None
    _providers = None
    _swap: Tuple = (2, 0, 1)
    _h_index: int = 2
    _w_index: int = 3
    _norm_shape: List = [1,3,1,1]

    # onnx
    _onnx_dtypes_to_np_dtypes = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
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
        mean: Optional[np.ndarray] = np.array([0.000, 0.000, 0.000], dtype=np.float32),
        std: Optional[np.ndarray] = np.array([1.000, 1.000, 1.000], dtype=np.float32),
    ):
        self._runtime = runtime
        self._model_path = model_path
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
            self._norm_shape = [1,3,1,1]

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
            self._norm_shape = [1,1,1,3]

        self._mean = mean.reshape(self._norm_shape)
        self._std = std.reshape(self._norm_shape)

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
        raise NotImplementedError()

class OSNet(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'osnet_ain_d_m_c_1Nx3x256x128.onnx',
        providers: Optional[List] = None,
    ):
        """OSNet

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for OSNet. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for OSNet

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            providers=providers,
            mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
            std=np.array([0.229, 0.224, 0.225], dtype=np.float32),
        )

    def __call__(
        self,
        *,
        base_image: np.ndarray,
        target_images: List[np.ndarray],
    ) -> np.ndarray:
        """OSNet

        Parameters
        ----------
        base_image: np.ndarray
            Entire image

        target_image: np.ndarray
            Entire image

        Returns
        -------
        similarity: np.ndarray
            similarity
        """
        temp_base_image = copy.deepcopy(base_image)
        temp_target_images = copy.deepcopy(target_images)

        # PreProcess
        stacked_images = \
            self._preprocess(
                base_image=temp_base_image,
                target_images=temp_target_images,
            )

        # Inference
        outputs = super().__call__(input_datas=[stacked_images])
        similarity = outputs[0]
        return similarity

    def _preprocess(
        self,
        *,
        base_image: np.ndarray,
        target_images: List[np.ndarray],
    ) -> Tuple[np.ndarray, int, int]:
        """_preprocess

        Parameters
        ----------
        base_image: np.ndarray
            Entire image

        target_image: List[np.ndarray]
            Entire image

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        stacked_images: np.ndarray
            Resized and normalized image. [1+N, 3, H, W]
        """
        # Resize + Transpose
        resized_base_image: np.ndarray = \
            cv2.resize(
                src=base_image,
                dsize=(
                    int(self._input_shapes[0][self._w_index]),
                    int(self._input_shapes[0][self._h_index]),
                )
            )
        resized_base_image = resized_base_image[..., ::-1]
        resized_base_image = resized_base_image.transpose(self._swap)

        resized_target_images_np: np.ndarray = None
        resized_target_images_list: List[np.ndarray] = []
        for target_image in target_images:
            resized_target_image: np.ndarray = \
                cv2.resize(
                    src=target_image,
                    dsize=(
                        int(self._input_shapes[0][self._w_index]),
                        int(self._input_shapes[0][self._h_index]),
                    )
                )
            resized_target_image = resized_target_image[..., ::-1]
            resized_target_image = resized_target_image.transpose(self._swap)
            resized_target_images_list.append(resized_target_image)
        resized_target_images_np = np.asarray(resized_target_images_list)

        stacked_images = \
            np.vstack(
                [
                    resized_base_image[np.newaxis, ...],
                    resized_target_images_np,
                ]
            )
        stacked_images = (stacked_images / 255.0 - self._mean) / self._std
        stacked_images = stacked_images.astype(self._input_dtypes[0])
        return stacked_images


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
        default='osnet_ain_d_m_c_1Nx3x256x128.onnx',
        help='ONNX/TFLite file path for OSNet.',
    )
    parser.add_argument(
        '-bi',
        '--base_image',
        type=str,
        help='Base image file.',
    )
    parser.add_argument(
        '-tis',
        '--target_images',
        type=str,
        nargs='+',
        help='Target image files.',
    )
    parser.add_argument(
        '-ep',
        '--execution_provider',
        type=str,
        choices=['cpu', 'cuda', 'tensorrt'],
        default='cpu',
        help='Execution provider for ONNXRuntime.',
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
    base_image_file: str = args.base_image
    target_image_files: str = args.target_images
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
    model = OSNet(
        runtime=runtime,
        model_path=model_file,
        providers=providers,
    )

    target_images: List[np.ndarray] = []

    base_image: np.ndarray = cv2.imread(base_image_file)

    for target_image_file in target_image_files:
        target_image: np.ndarray = cv2.imread(target_image_file)
        target_images.append(target_image)

    start_time = time.perf_counter()
    similarities = \
        model(
            base_image=base_image,
            target_images=target_images,
        )
    elapsed_time = time.perf_counter() - start_time

    for idx, n in enumerate(similarities):
        similarity_str = ""
        for m in n:
            similarity_str = f"{similarity_str}{m:.3f} "
        print(f'Base.{idx+1}: {Color.GREEN("The similarities are")} {similarity_str}')
    print(f'{Color.GREEN("Elapsed time:")} {elapsed_time*1000:.2f}ms' )

if __name__ == "__main__":
    main()

"""
# 1.png vs 2.png
The similarity is 0.511 Elapsed time: 157.37ms
# 1.png vs 3.png
The similarity is 0.764 Elapsed time: 157.37ms
# 1.png vs 4.png
The similarity is 0.725 Elapsed time: 158.05ms

# CPU: 1.png vs 2.png, 3.png, 4.png
Base.1: The similarities are 0.511 0.764 0.725
Elapsed time: 213.46ms

# CUDA: 1.png vs 2.png, 3.png, 4.png
Base.1: The similarities are 0.511 0.764 0.725
Elapsed time: 113.07ms
"""