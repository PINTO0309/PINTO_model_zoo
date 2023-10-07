import onnxruntime as ort
from collections import OrderedDict

MODELS = OrderedDict(
    {
        "01": "gold_yolo_n_hand_0423_0.2295_1x3x192x320.onnx",
        "02": "gold_yolo_n_hand_0423_0.2295_1x3x192x416.onnx",
        "03": "gold_yolo_n_hand_0423_0.2295_1x3x192x640.onnx",
        "04": "gold_yolo_n_hand_0423_0.2295_1x3x192x800.onnx",
        "05": "gold_yolo_n_hand_0423_0.2295_1x3x256x320.onnx",
        "06": "gold_yolo_n_hand_0423_0.2295_1x3x256x416.onnx",
        "07": "gold_yolo_n_hand_0423_0.2295_1x3x256x640.onnx",
        "08": "gold_yolo_n_hand_0423_0.2295_1x3x256x800.onnx",
        "09": "gold_yolo_n_hand_0423_0.2295_1x3x256x960.onnx",
        "10": "gold_yolo_n_hand_0423_0.2295_1x3x288x1280.onnx",
        "11": "gold_yolo_n_hand_0423_0.2295_1x3x288x480.onnx",
        "12": "gold_yolo_n_hand_0423_0.2295_1x3x288x640.onnx",
        "13": "gold_yolo_n_hand_0423_0.2295_1x3x288x800.onnx",
        "14": "gold_yolo_n_hand_0423_0.2295_1x3x288x960.onnx",
        "15": "gold_yolo_n_hand_0423_0.2295_1x3x320x320.onnx",
        "16": "gold_yolo_n_hand_0423_0.2295_1x3x384x1280.onnx",
        "17": "gold_yolo_n_hand_0423_0.2295_1x3x384x480.onnx",
        "18": "gold_yolo_n_hand_0423_0.2295_1x3x384x640.onnx",
        "19": "gold_yolo_n_hand_0423_0.2295_1x3x384x800.onnx",
        "20": "gold_yolo_n_hand_0423_0.2295_1x3x384x960.onnx",
        "21": "gold_yolo_n_hand_0423_0.2295_1x3x416x416.onnx",
        "22": "gold_yolo_n_hand_0423_0.2295_1x3x480x1280.onnx",
        "23": "gold_yolo_n_hand_0423_0.2295_1x3x480x640.onnx",
        "24": "gold_yolo_n_hand_0423_0.2295_1x3x480x800.onnx",
        "25": "gold_yolo_n_hand_0423_0.2295_1x3x480x960.onnx",
        "26": "gold_yolo_n_hand_0423_0.2295_1x3x512x512.onnx",
        "27": "gold_yolo_n_hand_0423_0.2295_1x3x512x640.onnx",
        "28": "gold_yolo_n_hand_0423_0.2295_1x3x512x896.onnx",
        "29": "gold_yolo_n_hand_0423_0.2295_1x3x544x1280.onnx",
        "30": "gold_yolo_n_hand_0423_0.2295_1x3x544x800.onnx",
        "31": "gold_yolo_n_hand_0423_0.2295_1x3x544x960.onnx",
        "32": "gold_yolo_n_hand_0423_0.2295_1x3x640x640.onnx",
        "33": "gold_yolo_n_hand_0423_0.2295_1x3x736x1280.onnx",
    }
)

box_sizes = []
for k, v in MODELS.items():
    onnx_session = ort.InferenceSession(
        path_or_bytes=v,
        providers=['CPUExecutionProvider'],
    )
    box_sizes.append([onnx_session.get_inputs()[0].shape[2], onnx_session.get_inputs()[0].shape[3], onnx_session.get_outputs()[0].shape[1]])

print(f'MODELS count: {len(MODELS)}')
print(f'BOX_SIZE count: {len(box_sizes)}')

for box_size in box_sizes:
    print(f'"{box_size[0]} {box_size[1]} {box_size[2]}"')