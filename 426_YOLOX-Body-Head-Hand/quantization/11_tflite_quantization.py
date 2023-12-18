import numpy as np
import tensorflow as tf

"""
onnx2tf -i yolox_ti_body_head_hand_s_post_1x3x128x160.onnx -coion -osd
onnx2tf -i yolox_ti_body_head_hand_s_post_1x3x256x320.onnx -coion -osd
onnx2tf -i yolox_ti_body_head_hand_s_post_1x3x480x640.onnx -coion -osd
"""

RESOLUTIONS = [
    [128,160],
    [256,320],
    [480,640],
]

def representative_dataset_128x160():
    images = np.load('calibdata_bgr_no_norm_128x160.npy')
    for image in images:
        yield {
            "input": image[np.newaxis, ...],
        }

def representative_dataset_256x320():
    images = np.load('calibdata_bgr_no_norm_256x320.npy')
    for image in images:
        yield {
            "input": image[np.newaxis, ...],
        }

def representative_dataset_480x640():
    images = np.load('calibdata_bgr_no_norm_480x640.npy')
    for image in images:
        yield {
            "input": image[np.newaxis, ...],
        }


for H, W in RESOLUTIONS:
    print(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ {H}x{W}')
    converter = tf.lite.TFLiteConverter.from_saved_model(f'saved_model_{H}x{W}')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if H == 128:
        converter.representative_dataset = representative_dataset_128x160
    elif H == 256:
        converter.representative_dataset = representative_dataset_256x320
    elif H == 480:
        converter.representative_dataset = representative_dataset_480x640
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    with open(f'saved_model_{H}x{W}/yolox_ti_body_head_hand_s_post_1x3x{H}x{W}_bgr_uint8.tflite', 'wb') as w:
        w.write(tflite_quant_model)

"""
tfliteiorewriter -i saved_model_128x160/yolox_ti_body_head_hand_s_post_1x3x128x160_bgr_uint8.tflite
tfliteiorewriter -i saved_model_256x320/yolox_ti_body_head_hand_s_post_1x3x256x320_bgr_uint8.tflite
tfliteiorewriter -i saved_model_480x640/yolox_ti_body_head_hand_s_post_1x3x480x640_bgr_uint8.tflite
"""