import os
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# ONNXモデルのパス
# MODEL_PATH = "bodyparse_lip_20_1x3x480x640.onnx"
# MODEL_PATH = "bodyparse_atr_18_1x3x480x640.onnx"
MODEL_PATH = "bodyparse_pascal_7_1x3x480x640.onnx"
INPUT_IMAGE_PATH = "inputs"  # 画像フォルダ
OUTPUT_FOLDER = "outputs_onnx"

# 出力フォルダの作成
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ONNXセッションの作成
session_option = ort.SessionOptions()
session_option.log_severity_level = 3
sess = ort.InferenceSession(
    path_or_bytes=MODEL_PATH,
    sess_options=session_option,
    providers=[
        'CUDAExecutionProvider',
        'CPUExecutionProvider',
    ]
)
input_name = sess.get_inputs()[0].name
n, c, h, w = sess.get_inputs()[0].shape
output_name = sess.get_outputs()[0].name

# カラーマップ
COLOR_MAP = np.array([
    [0, 0, 0],        # 背景 (黒)
    [255, 0, 0],      # クラス1 (赤)
    [0, 255, 0],      # クラス2 (緑)
    [0, 0, 255],      # クラス3 (青)
    [255, 255, 0],    # クラス4 (黄)
    [255, 0, 255],    # クラス5 (マゼンタ)
    [0, 255, 255],    # クラス6 (シアン)
    [128, 0, 0],      # クラス7 (ダークレッド)
    [0, 128, 0],      # クラス8 (ダークグリーン)
    [0, 0, 128],      # クラス9 (ダークブルー)
    [128, 128, 0],    # クラス10 (オリーブ)
    [128, 0, 128],    # クラス11 (紫)
    [0, 128, 128],    # クラス12 (ティール)
    [192, 192, 192],  # クラス13 (シルバー)
    [128, 128, 128],  # クラス14 (グレー)
    [64, 64, 64],     # クラス15 (ダークグレー)
    [255, 165, 0],    # クラス16 (オレンジ)
    [75, 0, 130],     # クラス17 (インディゴ)
    [199, 21, 133],   # クラス18 (バイオレット)
    [144, 238, 144]   # クラス19 (ライトグリーン)
], dtype=np.uint8)

mean = np.asarray([0.406, 0.456, 0.485], dtype=np.float32)
std = np.asarray([0.225, 0.224, 0.229], dtype=np.float32)

# 画像の前処理関数
def preprocess_image(img: np.ndarray):
    img = cv2.resize(img, (w, h))
    img = img.astype(np.float32) / 255.0  # 0-1に正規化
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # [H, W, C] → [C, H, W]
    img = np.expand_dims(img, axis=0)  # [1, 3, 480, 640]
    return img

# 推論と後処理
def run_inference(image_path):
    orig_img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGRで読み込む
    orig_h, orig_w = orig_img.shape[0], orig_img.shape[1]
    img = preprocess_image(orig_img)

    # ONNXモデルで推論
    result = sess.run([output_name], {input_name: img})[0]  # (1,1,480,640)

    # 出力データを 480x640 の numpy array に変換
    mask = result[0, 0, :, :].astype(np.uint8)  # (480, 640)

    # カラーマップ適用
    color_mask = COLOR_MAP[mask]  # (480, 640, 3)

    # マスクのリサイズ
    color_mask = cv2.resize(color_mask, (orig_w, orig_h))

    # マスクを重ねる (透過度 0.5)
    overlay = cv2.addWeighted(orig_img, 0.6, color_mask, 0.4, 0)

    # 保存
    output_path = os.path.join(OUTPUT_FOLDER, f"{os.path.basename(image_path)}")
    cv2.imwrite(output_path, overlay)

# 指定フォルダ内の画像を処理
for file in tqdm(os.listdir(INPUT_IMAGE_PATH), dynamic_ncols=True):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(INPUT_IMAGE_PATH, file)
        run_inference(image_path)
