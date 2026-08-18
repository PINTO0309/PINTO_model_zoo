#!/usr/bin/env python

from __future__ import annotations
import os
import copy
import cv2
from tqdm import tqdm
import numpy as np
from enum import Enum
from typing import Tuple, Optional, List, Dict
import importlib.util
from abc import ABC, abstractmethod

MODELS = [
    "mlfn-9cb5a267_NMx3x256x128.onnx",
    "mobilenetv2_1dot0_duke_NMx3x256x128.onnx",
    "mobilenetv2_1dot0_market_NMx3x256x128.onnx",
    "mobilenetv2_1dot0_msmt_NMx3x256x128.onnx",
    "mobilenetv2_1dot4_duke_NMx3x256x128.onnx",
    "mobilenetv2_1dot4_market_NMx3x256x128.onnx",
    "mobilenetv2_1dot4_msmt_NMx3x256x128.onnx",
    "mobilenetv2_1_NMx3x256x128.onnx",
    "osnet_ain_d_m_c_NMx3x256x128.onnx",
    "osnet_ain_ms_d_c_NMx3x256x128.onnx",
    "osnet_ain_ms_d_m_NMx3x256x128.onnx",
    "osnet_ain_ms_m_c_NMx3x256x128.onnx",
    "osnet_ain_x0_25_imagenet_NMx3x256x128.onnx",
    "osnet_ain_x0_5_imagenet_NMx3x256x128.onnx",
    "osnet_ain_x0_75_imagenet_NMx3x256x128.onnx",
    "osnet_ain_x1_0_dukemtmcreid_256x128_amsgrad_ep90_lr0_NMx3x256x128.onnx",
    "osnet_ain_x1_0_imagenet_NMx3x256x128.onnx",
    "osnet_ain_x1_0_market1501_256x128_amsgrad_ep100_lr0_NMx3x256x128.onnx",
    "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0_NMx3x256x128.onnx",
    "osnet_d_m_c_NMx3x256x128.onnx",
    "osnet_ibn_d_m_c_NMx3x256x128.onnx",
    "osnet_ibn_ms_d_c_NMx3x256x128.onnx",
    "osnet_ibn_ms_d_m_NMx3x256x128.onnx",
    "osnet_ibn_ms_m_c_NMx3x256x128.onnx",
    "osnet_ibn_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0_NMx3x256x128.onnx",
    "osnet_ibn_x1_0_imagenet_NMx3x256x128.onnx",
    "osnet_ibn_x1_0_market1501_256x128_amsgrad_ep150_stp60_lr0_NMx3x256x128.onnx",
    "osnet_ibn_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_NMx3x256x128.onnx",
    "osnet_ms_d_c_NMx3x256x128.onnx",
    "osnet_ms_d_m_NMx3x256x128.onnx",
    "osnet_ms_m_c_NMx3x256x128.onnx",
    "osnet_x0_25_duke_256x128_amsgrad_ep180_stp80_lr0_NMx3x256x128.onnx",
    "osnet_x0_25_imagenet_NMx3x256x128.onnx",
    "osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0_NMx3x256x128.onnx",
    "osnet_x0_25_msmt17_256x128_amsgrad_ep180_stp80_lr0_NMx3x256x128.onnx",
    "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_NMx3x256x128.onnx",
    "osnet_x0_5_duke_256x128_amsgrad_ep180_stp80_lr0_NMx3x256x128.onnx",
    "osnet_x0_5_imagenet_NMx3x256x128.onnx",
    "osnet_x0_5_market_256x128_amsgrad_ep180_stp80_lr0_NMx3x256x128.onnx",
    "osnet_x0_5_msmt17_256x128_amsgrad_ep180_stp80_lr0_NMx3x256x128.onnx",
    "osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_NMx3x256x128.onnx",
    "osnet_x0_75_duke_256x128_amsgrad_ep150_stp60_lr0_NMx3x256x128.onnx",
    "osnet_x0_75_imagenet_NMx3x256x128.onnx",
    "osnet_x0_75_market_256x128_amsgrad_ep150_stp60_lr0_NMx3x256x128.onnx",
    "osnet_x0_75_msmt17_256x128_amsgrad_ep150_stp60_lr0_NMx3x256x128.onnx",
    "osnet_x0_75_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_NMx3x256x128.onnx",
    "osnet_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0_NMx3x256x128.onnx",
    "osnet_x1_0_imagenet_NMx3x256x128.onnx",
    "osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0_NMx3x256x128.onnx",
    "osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0_NMx3x256x128.onnx",
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_NMx3x256x128.onnx",
    "resnet50_fc512_msmt_xent_NMx3x256x128.onnx",
    "resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_NMx3x256x128.onnx",
]

# Single-input models that output L2-normalized embeddings.
# https://github.com/PINTO0309/PersonViT
#   Input: images [N, 3, 256, 128], RGB, [0, 1] scaling, mean=0.5, std=0.5
#   Output: embeddings [N, D] (L2 normalized)
UNIFIED_MODELS = [
    "osnet_x1_0_p_unified_n.onnx",
    "personvit_vitb16_ain_unified_aug_n.onnx",
    "personvit_vits16_ain_unified_aug_n.onnx",
]

# NOTE: the OSNet unified model outputs embeddings compressed into a narrow
# cosine cone (different-person pairs still score ~0.7), so read its row by
# the same/different ORDERING, not by absolute values - thresholds tuned for
# the other models do not transfer. In deployment the cone is expanded with
# embedding whitening, e' = normalize((e - mu) / sd), where mu/sd are running
# statistics over a large multi-identity pool and the decision thresholds are
# re-derived afterwards (ptrack exp050/exp060). The six test crops here span
# only two identities - far too few for those statistics - so no whitening is
# applied in this script.

# [base_image_file, target_image_file]
TEST_IMAGES = [
    ["00030.jpg", "00031.jpg"], # ↓
    ["00030.jpg", "1.png"], # ↓
    ["00031.jpg", "2.png"], # ↓
    ["1.png", "2.png"], # ↑
    ["1.png", "3.png"], # ↑
    ["1.png", "4.png"], # ↑
]

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
        model_path: Optional[str] = 'osnet_ain_d_m_c_NMx3x256x128.onnx',
        providers: Optional[List] = None,
        mean: Optional[np.ndarray] = np.array([0.485, 0.456, 0.406], dtype=np.float32),
        std: Optional[np.ndarray] = np.array([0.229, 0.224, 0.225], dtype=np.float32),
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

        mean: Optional[np.ndarray]
            Normalization mean. Default: ImageNet mean

        std: Optional[np.ndarray]
            Normalization std. Default: ImageNet std
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            providers=providers,
            mean=mean,
            std=std,
        )

    def __call__(
        self,
        *,
        base_images: List[np.ndarray],
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
        temp_base_images = copy.deepcopy(base_images)
        temp_target_images = copy.deepcopy(target_images)

        # PreProcess
        base_images, target_images = \
            self._preprocess(
                base_images=temp_base_images,
                target_images=temp_target_images,
            )

        # Inference
        outputs = super().__call__(input_datas=[base_images, target_images])
        similarity = outputs[0]
        return similarity

    def _preprocess(
        self,
        *,
        base_images: List[np.ndarray],
        target_images: List[np.ndarray],
    ) -> Tuple[np.ndarray, int, int]:
        """_preprocess

        Parameters
        ----------
        base_images: List[np.ndarray]
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
        stacked_images_N: np.ndarray
            Resized and normalized image. [N, 3, H, W]

        stacked_images_M: np.ndarray
            Resized and normalized image. [M, 3, H, W]
        """
        # Resize + Transpose
        resized_base_images_np: np.ndarray = None
        resized_base_images_list: List[np.ndarray] = []
        for base_image in base_images:
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
            resized_base_images_list.append(resized_base_image)
        resized_base_images_np = np.asarray(resized_base_images_list)
        base_images_num = len(resized_base_images_np)

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
        target_images_num = len(resized_target_images_np)

        stacked_images = \
            np.vstack(
                [
                    resized_base_images_np,
                    resized_target_images_np,
                ]
            )
        stacked_images = (stacked_images / 255.0 - self._mean) / self._std
        stacked_images = stacked_images.astype(self._input_dtypes[0])
        return stacked_images[0:base_images_num, ...], stacked_images[base_images_num:base_images_num+target_images_num, ...]

class UnifiedReID(OSNet):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'osnet_x1_0_p_unified_n.onnx',
        providers: Optional[List] = None,
    ):
        """UnifiedReID

        Single-input ReID models that output L2-normalized embeddings.
        https://github.com/PINTO0309/PersonViT

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for UnifiedReID. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for UnifiedReID

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            providers=providers,
            mean=np.array([0.500, 0.500, 0.500], dtype=np.float32),
            std=np.array([0.500, 0.500, 0.500], dtype=np.float32),
        )

    def __call__(
        self,
        *,
        base_images: List[np.ndarray],
        target_images: List[np.ndarray],
    ) -> np.ndarray:
        """UnifiedReID

        Parameters
        ----------
        base_images: List[np.ndarray]
            Entire image

        target_images: List[np.ndarray]
            Entire image

        Returns
        -------
        similarity: np.ndarray
            Cosine similarity. [N, M]
        """
        base_embeddings = self.embed(images=base_images)
        target_embeddings = self.embed(images=target_images)
        similarity = base_embeddings @ target_embeddings.T
        return similarity

    def embed(
        self,
        *,
        images: List[np.ndarray],
    ) -> np.ndarray:
        """embed

        Parameters
        ----------
        images: List[np.ndarray]
            Entire image

        Returns
        -------
        embeddings: np.ndarray
            L2-normalized embeddings. [N, D]
        """
        temp_images = copy.deepcopy(images)

        # PreProcess (Resize + BGR->RGB + Transpose + Normalize)
        resized_images_list: List[np.ndarray] = []
        for image in temp_images:
            resized_image: np.ndarray = \
                cv2.resize(
                    src=image,
                    dsize=(
                        int(self._input_shapes[0][self._w_index]),
                        int(self._input_shapes[0][self._h_index]),
                    )
                )
            resized_image = resized_image[..., ::-1]
            resized_image = resized_image.transpose(self._swap)
            resized_images_list.append(resized_image)
        images_np = np.asarray(resized_images_list)
        images_np = (images_np / 255.0 - self._mean) / self._std
        images_np = images_np.astype(self._input_dtypes[0])

        # Inference
        outputs = AbstractModel.__call__(self, input_datas=[images_np])
        embeddings: np.ndarray = outputs[0]

        # Output embeddings are L2 normalized, but re-normalize just in case.
        embeddings = embeddings \
            / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)
        return embeddings

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
    md_str = ""
    md_str = md_str + "|Model|00030 vs 00031|00030 vs 1|00031 vs 2|1 vs 2|1 vs 3|1 vs 4|\n"
    md_str = md_str + "|:-|-:|-:|-:|-:|-:|-:|\n"

    for model_file in tqdm(MODELS + UNIFIED_MODELS):
        if not os.path.isfile(model_file):
            tqdm.write(f'{Color.YELLOW("Skip (not found):")} {model_file}')
            continue

        model_class = UnifiedReID if model_file in UNIFIED_MODELS else OSNet
        model = model_class(
            runtime='onnx',
            model_path=model_file,
            providers=[
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ],
        )

        if model_file in UNIFIED_MODELS:
            # Embed every unique test image once, then score the pairs.
            image_files: List[str] = []
            for base_image_file, target_image_file in TEST_IMAGES:
                for image_file in [base_image_file, target_image_file]:
                    if image_file not in image_files:
                        image_files.append(image_file)
            embeddings = \
                model.embed(
                    images=[cv2.imread(image_file) for image_file in image_files],
                )

            def pair_sims(embs: np.ndarray) -> List[str]:
                embedding_by_file: Dict[str, np.ndarray] = {
                    image_file: embedding \
                        for image_file, embedding in zip(image_files, embs)
                }
                return [
                    f'{float(embedding_by_file[b] @ embedding_by_file[t]):.3f}' \
                        for b, t in TEST_IMAGES
                ]

            sims = pair_sims(embeddings)
            md_str = md_str + f"|{model_file}|{sims[0]}|{sims[1]}|{sims[2]}|{sims[3]}|{sims[4]}|{sims[5]}|\n"
        else:
            sims: List[str] = []
            for base_image_file, target_image_file in TEST_IMAGES:
                base_image: np.ndarray = cv2.imread(base_image_file)
                target_image: np.ndarray = cv2.imread(target_image_file)

                similarities = \
                    model(
                        base_images=[base_image],
                        target_images=[target_image],
                    )
                sims.append(f'{float(np.squeeze(similarities)):.3f}')
            md_str = md_str + f"|{model_file}|{sims[0]}|{sims[1]}|{sims[2]}|{sims[3]}|{sims[4]}|{sims[5]}|\n"
    print(md_str)

if __name__ == "__main__":
    main()
