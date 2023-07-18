import os
import shutil
import glob
import requests
import shutil
import torch
import torch.nn.functional as F
from models.model_registry import model_entrypoint
import onnx2tf
import tf2onnx
import onnx
from onnxsim import simplify

MODELS = [
    "ffnet101.zip",
    "ffnet122N.zip",
    "ffnet122NS.zip",
    "ffnet134.zip",
    "ffnet150.zip",
    "ffnet150S.zip",
    "ffnet18.zip",
    "ffnet34.zip",
    "ffnet40S.zip",
    "ffnet46N.zip",
    "ffnet46NS.zip",
    "ffnet50.zip",
    "ffnet54S.zip",
    "ffnet56.zip",
    "ffnet74N.zip",
    "ffnet74NS.zip",
    "ffnet78S.zip",
    "ffnet86.zip",
    "ffnet86S.zip",
]

download_path = os.path.join(".", "model_weights")
os.makedirs(download_path, exist_ok=True)

for model in MODELS:
    download_url = "https://github.com/Qualcomm-AI-research/FFNet/releases/download/models/" + model
    file_path = os.path.join(".", "model_weights", model)
    urlData = requests.get(download_url).content
    with open(file_path ,mode='wb') as f:
        f.write(urlData)
    shutil.unpack_archive(file_path, download_path)
    os.remove(file_path)

SEG_MODEL_NAME = {
    ("segmentation_ffnet101_AAA", (1024, 2048)),
    ("segmentation_ffnet50_AAA", (1024, 2048)),
    ("segmentation_ffnet150_AAA", (1024, 2048)),
    ("segmentation_ffnet134_AAA", (1024, 2048)),
    ("segmentation_ffnet86_AAA", (1024, 2048)),
    ("segmentation_ffnet56_AAA", (1024, 2048)),
    ("segmentation_ffnet34_AAA", (1024, 2048)),
    ("segmentation_ffnet150_ABB", (1024, 2048)),
    ("segmentation_ffnet86_ABB", (1024, 2048)),
    ("segmentation_ffnet56_ABB", (1024, 2048)),
    ("segmentation_ffnet34_ABB", (1024, 2048)),
    ("segmentation_ffnet150S_BBB", (1024, 2048)),
    ("segmentation_ffnet86S_BBB", (1024, 2048)),
    ("segmentation_ffnet122N_CBB", (1024, 2048)),
    ("segmentation_ffnet74N_CBB", (1024, 2048)),
    ("segmentation_ffnet46N_CBB", (1024, 2048)),
    ("segmentation_ffnet101_dAAA", (1024, 2048)),
    ("segmentation_ffnet50_dAAA", (1024, 2048)),
    ("segmentation_ffnet150_dAAA", (1024, 2048)),
    ("segmentation_ffnet134_dAAA", (1024, 2048)),
    ("segmentation_ffnet86_dAAA", (1024, 2048)),
    ("segmentation_ffnet56_dAAA", (1024, 2048)),
    ("segmentation_ffnet34_dAAA", (1024, 2048)),
    ("segmentation_ffnet18_dAAA", (1024, 2048)),
    # ("segmentation_ffnet150_dAAC", (1024, 2048)), size mismatch
    # ("segmentation_ffnet86_dAAC", (512, 1024)), size mismatch
    # ("segmentation_ffnet34_dAAC", (512, 1024)), size mismatch
    # ("segmentation_ffnet18_dAAC", (512, 1024)), size mismatch
    ("segmentation_ffnet150S_dBBB", (1024, 2048)),
    ("segmentation_ffnet86S_dBBB", (1024, 2048)),
    ("segmentation_ffnet86S_dBBB_mobile", (1024, 2048)),
    ("segmentation_ffnet78S_dBBB_mobile", (1024, 2048)),
    ("segmentation_ffnet54S_dBBB_mobile", (1024, 2048)),
    ("segmentation_ffnet40S_dBBB_mobile", (1024, 2048)),
    ("segmentation_ffnet150S_BBB_mobile", (512, 1024)),
    ("segmentation_ffnet150S_BBB_mobile_pre_down", (512, 1024)),
    ("segmentation_ffnet86S_BBB_mobile", (512, 1024)),
    ("segmentation_ffnet86S_BBB_mobile_pre_down", (512, 1024)),
    ("segmentation_ffnet78S_BBB_mobile", (512, 1024)),
    ("segmentation_ffnet78S_BBB_mobile_pre_down", (512, 1024)),
    ("segmentation_ffnet54S_BBB_mobile", (512, 1024)),
    ("segmentation_ffnet54S_BBB_mobile_pre_down", (512, 1024)),
    ("segmentation_ffnet40S_BBB_mobile", (512, 1024)),
    ("segmentation_ffnet40S_BBB_mobile_pre_down", (512, 1024)),
    ("segmentation_ffnet150S_BCC_mobile", (512, 1024)),
    ("segmentation_ffnet40S_BBB_mobile_pre_down", (512, 1024)),
    ("segmentation_ffnet40S_BBB_mobile", (512, 1024)),
    ("segmentation_ffnet86S_BCC_mobile_pre_down", (512, 1024)),
    ("segmentation_ffnet86S_BCC_mobile", (512, 1024)),
    ("segmentation_ffnet78S_BCC_mobile_pre_down", (512, 1024)),
    ("segmentation_ffnet78S_BCC_mobile", (512, 1024)),
    ("segmentation_ffnet54S_BCC_mobile_pre_down", (512, 1024)),
    ("segmentation_ffnet54S_BCC_mobile", (512, 1024)),
    ("segmentation_ffnet40S_BCC_mobile_pre_down", (512, 1024)),
    ("segmentation_ffnet40S_BCC_mobile", (512, 1024)),
    ("segmentation_ffnet122NS_CBB_mobile_pre_down", (512, 1024)),
    ("segmentation_ffnet122NS_CBB_mobile", (512, 1024)),
    ("segmentation_ffnet74NS_CBB_mobile_pre_down", (512, 1024)),
    ("segmentation_ffnet74NS_CBB_mobile", (512, 1024)),
    ("segmentation_ffnet46NS_CBB_mobile_pre_down", (512, 1024)),
    ("segmentation_ffnet46NS_CBB_mobile", (512, 1024)),
    ("segmentation_ffnet122NS_CCC_mobile_pre_down", (512, 1024)),
    ("segmentation_ffnet122NS_CCC_mobile", (512, 1024)),
    # ("segmentation_ffnet74NS_CCC_mobile_pre_down", (512, 1024)),  RuntimeError: Error(s) in loading state_dict for FFNet:
    ("segmentation_ffnet74NS_CCC_mobile", (512, 1024)),
    ("segmentation_ffnet46NS_CCC_mobile_pre_down", (512, 1024)),
    ("segmentation_ffnet46NS_CCC_mobile", (512, 1024))
}

class ExportFFNet(torch.nn.Module):
    def __init__(self, model_name, h, w):
        super().__init__()
        self.model = model_entrypoint(model_name)()
        self.h = h
        self.w = w

    def forward(self, x):
        x = self.model(x)
        x = F.interpolate(x, (self.h, self.w), mode="bilinear", align_corners=True)
        x = torch.argmax(x, dim=1)
        return x

output_path = os.path.join(".", "ffnet_models")

if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.mkdir(output_path)

if os.path.exists(os.path.join(".", "saved_model")):
    shutil.rmtree(os.path.join(".", "saved_model"))
for tmp_onnx_path in glob.glob(os.path.join(".", "*.onnx"), recursive=False):
    os.remove(tmp_onnx_path)

for model_name, size in SEG_MODEL_NAME:
    height, width = size

    print("----- Start {} -----".format(model_name), )

    if os.path.exists(os.path.join(output_path, model_name + "_fused_argmax.onnx")):
        print("{} has already been exported.".format(model_name))
        continue

    # load model and export onnx.
    model = ExportFFNet(model_name=model_name, h=height, w=width)
    dummy_input = torch.randn(1, 3, height, width, device="cpu")
    tmp_onnx_path = os.path.join(".", model_name + ".onnx")
    torch.onnx.export(
        model,
        dummy_input,
        tmp_onnx_path,
        verbose=False,
        input_names=[ "input1" ],
        output_names=[ "output1" ]
    )

    # Convert default argmax.
    output_onnx_path = os.path.join(output_path, model_name + ".onnx")
    tflite_float32_path = os.path.join(".", "saved_model", model_name + "_float32.tflite")

    onnx2tf.convert(
        input_onnx_file_path=tmp_onnx_path,
        non_verbose=True,
    )

    model_proto, external_tensor_storage = \
        tf2onnx.convert.from_tflite(
            tflite_path=tflite_float32_path,
            inputs_as_nchw=['inputs_0'],
            output_path=output_onnx_path,
        )
    model_onnx1 = onnx.load(output_onnx_path)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, output_onnx_path)
    model_onnx2 = onnx.load(output_onnx_path)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, output_onnx_path)
    model_onnx2 = onnx.load(output_onnx_path)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, output_onnx_path)
    model_onnx2 = onnx.load(output_onnx_path)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, output_onnx_path)

    for tflite_file in glob.glob(os.path.join(".", "saved_model", "*.tflite"), recursive=True):
        shutil.copyfile(tflite_file, os.path.join(output_path, os.path.basename(tflite_file)))
    shutil.rmtree(os.path.join(".", "saved_model"))

    # Convert fused-argmax
    src_name = tmp_onnx_path
    tmp_onnx_path = os.path.join(".", model_name + "_fused_argmax.onnx")
    os.rename(src_name, tmp_onnx_path)
    output_onnx_path = os.path.join(output_path, model_name + "_fused_argmax.onnx")
    tflite_float32_path = os.path.join(".", "saved_model", model_name + "_fused_argmax_float32.tflite")

    onnx2tf.convert(
        input_onnx_file_path=tmp_onnx_path,
        replace_argmax_to_fused_argmax_and_indicies_is_int64=True,
        non_verbose=True,
    )

    model_proto, external_tensor_storage = \
        tf2onnx.convert.from_tflite(
            tflite_path=tflite_float32_path,
            inputs_as_nchw=['inputs_0'],
            output_path=output_onnx_path,
        )
    model_onnx1 = onnx.load(output_onnx_path)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, output_onnx_path)
    model_onnx2 = onnx.load(output_onnx_path)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, output_onnx_path)
    model_onnx2 = onnx.load(output_onnx_path)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, output_onnx_path)
    model_onnx2 = onnx.load(output_onnx_path)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, output_onnx_path)

    for tflite_file in glob.glob(os.path.join(".", "saved_model", "*.tflite"), recursive=True):
        shutil.copyfile(tflite_file, os.path.join(output_path, os.path.basename(tflite_file)))
    os.remove(tmp_onnx_path)
    shutil.rmtree(os.path.join(".", "saved_model"))
