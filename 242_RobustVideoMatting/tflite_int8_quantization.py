import tensorflow as tf
import numpy as np
import itertools
import argparse

"""
    inputs['src'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 192, 320, 3)

    inputs['r1i'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 96, 160, 16)
        name: serving_default_r1i:0

    inputs['r2i'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 48, 80, 20)
        name: serving_default_r2i:0

    inputs['r3i'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 24, 40, 40)
        name: serving_default_r3i:0

    inputs['r4i'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 12, 20, 64)
        name: serving_default_r4i:0

successful patterns : ([1, 48, 80, 20], [1, 12, 20, 64], [1, 192, 320, 3], [1, 96, 160, 16], [1, 24, 40, 40])
"""

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='rvm_mobilenetv3', required=True)
parser.add_argument("--height", type=int, default=192, required=True)
parser.add_argument("--width", type=int, default=320, required=True)
args = parser.parse_args()

MODEL=args.model
H=args.height
W=args.width
saved_model_folder = f'{MODEL}_{H}x{W}'

raw_test_data = np.load('sample_npy/calibration_data_img_sample.npy')

shapes = None
if MODEL == 'rvm_mobilenetv3':
    shapes = [
        [1,H//4,W//4,20],
        [1,H//16,W//16,64],
        [1,H,W,3],
        [1,H//2,W//2,16],
        [1,H//8,W//8,40],
    ]
elif MODEL == 'rvm_resnet50':
    shapes = [
        [1,H//4,W//4,32],
        [1,H//16,W//16,128],
        [1,H,W,3],
        [1,H//2,W//2,16],
        [1,H//8,W//8,64],
    ]

permutations = itertools.permutations(shapes)

# Integer/Full Integer Quantization
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_folder)
for idx, datas in enumerate(permutations):
    try:
        def representative_dataset_gen():
            for idx in range(raw_test_data.shape[0]):
                image = raw_test_data[idx]
                calibdata = []

                for shape in datas:
                    if shape[3] == 3:
                        data = tf.image.resize(image, (shape[1], shape[2]))
                        data = data[np.newaxis,:,:,:]
                    else:
                        data = np.random.random_sample([i for i in shape]).astype(np.float32) * 255.0
                    tmp_data = eval('data / 255.0') # Default: (data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]
                    calibdata.append(tmp_data)
                yield calibdata

        print(f'@@@@@@@@@@@@@@@@@ Try.{idx+1}')

        # integer quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.representative_dataset = representative_dataset_gen
        tflite_model = converter.convert()
        with open(f'{saved_model_folder}/model_integer_quant.tflite', 'wb') as w:
            w.write(tflite_model)

        if MODEL == 'rvm_mobilenetv3':
            # full integer quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            inf_type = tf.uint8
            converter.inference_input_type = inf_type
            converter.inference_output_type = inf_type
            converter.representative_dataset = representative_dataset_gen
            tflite_model = converter.convert()
            with open(f'{saved_model_folder}/model_full_integer_quant.tflite', 'wb') as w:
                w.write(tflite_model)

        print(f'@@@@@@@@@@@@@@@@@ successful patterns : {datas}')
        break
    except:
        # import traceback
        # traceback.print_exc()
        print(f'@@@@@@@@@@@@@@@@@ failed : {datas}')

# Edgetpu
if MODEL == 'rvm_mobilenetv3':
    import subprocess
    result = subprocess.check_output(
        [
            'edgetpu_compiler',
            '-o', saved_model_folder,
            '-sad',
            f'{saved_model_folder}/model_full_integer_quant.tflite'
        ],
        stderr=subprocess.PIPE
    ).decode('utf-8')