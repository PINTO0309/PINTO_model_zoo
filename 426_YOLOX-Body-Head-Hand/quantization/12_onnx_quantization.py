from typing import Dict, Union
import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    quantize_static,
    QuantFormat,
    QuantType,
)

SIZE = "t"
RESOLUTIONS = [
    [128, 160],
    [256, 320],
    [480, 640],
]

"""
python -m onnxruntime.quantization.preprocess \
--input yolox_ti_body_head_hand_${SIZE}_1x3x${H}x${W}.onnx \
--output yolox_ti_body_head_hand_${SIZE}_1x3x${H}x${W}_prep.onnx
"""


class ImgDataReader(CalibrationDataReader):
    def __init__ (self, imgs: np.ndarray) -> None:
        imgs = imgs.transpose(0, 3, 1, 2)
        self.img_dicts = iter([{"input": img[np.newaxis, ...]} for img in imgs])

    def get_next(self) -> Union[Dict[str, np.ndarray], None]:
        return next(self.img_dicts, None)

for H, W in RESOLUTIONS:
    input_model_path = f'yolox_ti_body_head_hand_{SIZE}_1x3x{H}x{W}_prep.onnx'
    output_model_path = f'yolox_ti_body_head_hand_{SIZE}_1x3x{H}x{W}_uint8.onnx'
    calibration_data = np.load(f'calibdata_bgr_no_norm_{H}x{W}.npy')

    data_reader = ImgDataReader(calibration_data)

    # quant_format = QuantFormat.QDQ
    quant_format = QuantFormat.QOperator

    quantize_static(
        input_model_path,
        output_model_path,
        data_reader,
        quant_format=quant_format ,
        activation_type=QuantType.QUInt8,
    )

"""
rm \
yolox_ti_body_head_hand_*_1x3x128x160_prep.onnx \
yolox_ti_body_head_hand_*_1x3x256x320_prep.onnx \
yolox_ti_body_head_hand_*_1x3x480x640_prep.onnx
"""