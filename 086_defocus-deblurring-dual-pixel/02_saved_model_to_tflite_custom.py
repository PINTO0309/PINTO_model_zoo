#! /usr/bin/env python
### tensorflow==2.3.1
import os
import sys
import argparse
import numpy as np
from pathlib import Path
import re
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Color:
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
    REVERCE        = '\033[07m'
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

def convert(saved_model_dir_path,
            signature_def,
            input_shapes,
            model_output_dir_path,
            output_no_quant_float32_tflite,
            output_weight_quant_tflite,
            output_float16_quant_tflite,
            output_integer_quant_tflite,
            output_full_integer_quant_tflite,
            output_integer_quant_type,
            string_formulas_for_normalization,
            calib_ds_type,
            ds_name_for_tfds_for_calibration,
            split_name_for_tfds_for_calibration,
            download_dest_folder_path_for_the_calib_tfds,
            tfds_download_flg,
            output_tfjs,
            output_tftrt,
            output_coreml,
            output_edgetpu):

    print(f'{Color.REVERCE}Start conversion process from saved_model to tflite{Color.RESET}', '=' * 38)

    import subprocess
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

    # Load saved_model and change input shape
    # https://github.com/tensorflow/tensorflow/issues/30180#issuecomment-505959220
    model = tf.saved_model.load(saved_model_dir_path)
    if signature_def:
        concrete_func = model.signatures[signature_def]
    else:
        concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    if input_shapes:
        concrete_func_input_tensors = [tensor for tensor in concrete_func.inputs if tensor.dtype != tf.resource and not 'unknown' in tensor.name]
        for conc_input, def_input in zip(concrete_func_input_tensors, input_shapes):
            print('Before changing the input shape', conc_input)
            conc_input.set_shape(def_input)
            print('After  changing the input shape', conc_input)
    else:
        concrete_func_input_tensors = [tensor for tensor in concrete_func.inputs if tensor.dtype != tf.resource and not 'unknown' in tensor.name]
        for conc_input in concrete_func_input_tensors:
            input_shapes.append(conc_input.shape.as_list())

    # No Quantization - Input/Output=float32
    if output_no_quant_float32_tflite:
        try:
            print(f'{Color.REVERCE}tflite Float32 convertion started{Color.RESET}', '=' * 51)
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            tflite_model = converter.convert()
            with open(f'{model_output_dir_path}/model_float32.tflite', 'wb') as w:
                w.write(tflite_model)
            print(f'{Color.GREEN}tflite Float32 convertion complete!{Color.RESET} - {model_output_dir_path}/model_float32.tflite')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # Weight Quantization - Input/Output=float32
    if output_weight_quant_tflite:
        try:
            print(f'{Color.REVERCE}Weight Quantization started{Color.RESET}', '=' * 57)
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            tflite_model = converter.convert()
            with open(f'{model_output_dir_path}/model_weight_quant.tflite', 'wb') as w:
                w.write(tflite_model)
            print(f'{Color.GREEN}Weight Quantization complete!{Color.RESET} - {model_output_dir_path}/model_weight_quant.tflite')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # Float16 Quantization - Input/Output=float32
    if output_float16_quant_tflite:
        try:
            print(f'{Color.REVERCE}Float16 Quantization started{Color.RESET}', '=' * 56)
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            tflite_quant_model = converter.convert()
            with open(f'{model_output_dir_path}/model_float16_quant.tflite', 'wb') as w:
                w.write(tflite_quant_model)
            print(f'{Color.GREEN}Float16 Quantization complete!{Color.RESET} - {model_output_dir_path}/model_float16_quant.tflite')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # Downloading datasets for calibration
    raw_test_data = None
    if output_integer_quant_tflite or output_full_integer_quant_tflite:
        if calib_ds_type == 'tfds':
            print(f'{Color.REVERCE}TFDS download started{Color.RESET}', '=' * 63)
            raw_test_data = tfds.load(name=ds_name_for_tfds_for_calibration,
                                    with_info=False,
                                    split=split_name_for_tfds_for_calibration,
                                    data_dir=download_dest_folder_path_for_the_calib_tfds,
                                    download=tfds_download_flg)
            print(f'{Color.GREEN}TFDS download complete!{Color.RESET}')
        elif calib_ds_type == 'numpy':
            pass
        else:
            pass

    def representative_dataset_gen():
        for data in raw_test_data.take(10):
            image = data['image'].numpy()
            images = []
            for shape in input_shapes:
                data = tf.image.resize(image, (shape[1], shape[2]))
                tmp_image = eval(string_formulas_for_normalization) # Default: (data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]
                tmp_image = tf.repeat(tmp_image[np.newaxis,:,:,:], 2, axis=-1)
                print('@@@@@@@@@', tmp_image.shape)
                images.append(tmp_image)
            yield images

    # Integer Quantization
    if output_integer_quant_tflite:
        try:
            print(f'{Color.REVERCE}Integer Quantization started{Color.RESET}', '=' * 56)
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
            converter.representative_dataset = representative_dataset_gen
            tflite_model = converter.convert()
            with open(f'{model_output_dir_path}/model_integer_quant.tflite', 'wb') as w:
                w.write(tflite_model)
            print(f'{Color.GREEN}Integer Quantization complete!{Color.RESET} - {model_output_dir_path}/model_integer_quant.tflite')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # Full Integer Quantization
    if output_full_integer_quant_tflite:
        try:
            print(f'{Color.REVERCE}Full Integer Quantization started{Color.RESET}', '=' * 51)
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
            inf_type = None
            if output_integer_quant_type == 'int8':
                inf_type = tf.int8
            elif output_integer_quant_type == 'uint8':
                inf_type = tf.uint8
            else:
                inf_type = tf.int8
            converter.inference_input_type = inf_type
            converter.inference_output_type = inf_type
            converter.representative_dataset = representative_dataset_gen
            tflite_model = converter.convert()
            with open(f'{model_output_dir_path}/model_full_integer_quant.tflite', 'wb') as w:
                w.write(tflite_model)
            print(f'{Color.GREEN}Full Integer Quantization complete!{Color.RESET} - {model_output_dir_path}/model_full_integer_quant.tflite')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # EdgeTPU convert
    if output_edgetpu:
        try:
            print(f'{Color.REVERCE}EdgeTPU convertion started{Color.RESET}', '=' * 58)
            result = subprocess.check_output(['edgetpu_compiler',
                                              '-o', model_output_dir_path,
                                              '-s',
                                              f'{model_output_dir_path}/model_full_integer_quant.tflite'],
                                              stderr=subprocess.PIPE).decode('utf-8')
            print(result)
            print(f'{Color.GREEN}EdgeTPU convert complete!{Color.RESET} - {model_output_dir_path}/model_full_integer_quant_edgetpu.tflite')
        except subprocess.CalledProcessError as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
            import traceback
            traceback.print_exc()
            print("-" * 80)
            print('Please install edgetpu_compiler according to the following website.')
            print('https://coral.ai/docs/edgetpu/compiler/#system-requirements')

    # TensorFlow.js convert
    if output_tfjs:
        import subprocess
        try:
            print(f'{Color.REVERCE}TensorFlow.js Float32 convertion started{Color.RESET}', '=' * 44)
            result = subprocess.check_output(['tensorflowjs_converter',
                                            '--input_format', 'tf_saved_model',
                                            '--output_format', 'tfjs_graph_model',
                                            '--signature_name', 'serving_default',
                                            '--saved_model_tags', 'serve',
                                            saved_model_dir_path, f'{model_output_dir_path}/tfjs_model_float32'],
                                            stderr=subprocess.PIPE).decode('utf-8')
            print(result)
            print(f'{Color.GREEN}TensorFlow.js convertion complete!{Color.RESET} - {model_output_dir_path}/tfjs_model_float32')
        except subprocess.CalledProcessError as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
            import traceback
            traceback.print_exc()
        try:
            print(f'{Color.REVERCE}TensorFlow.js Float16 convertion started{Color.RESET}', '=' * 44)
            result = subprocess.check_output(['tensorflowjs_converter',
                                            '--quantize_float16',
                                            '--input_format', 'tf_saved_model',
                                            '--output_format', 'tfjs_graph_model',
                                            '--signature_name', 'serving_default',
                                            '--saved_model_tags', 'serve',
                                            saved_model_dir_path, f'{model_output_dir_path}/tfjs_model_float16'],
                                            stderr=subprocess.PIPE).decode('utf-8')
            print(result)
            print(f'{Color.GREEN}TensorFlow.js convertion complete!{Color.RESET} - {model_output_dir_path}/tfjs_model_float16')
        except subprocess.CalledProcessError as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
            import traceback
            traceback.print_exc()

    # TF-TRT (TensorRT) convert
    if output_tftrt:
        try:
            def input_fn():
                input_shapes_tmp = []
                for tf_input in input_shapes:
                    input_shapes_tmp.append(np.zeros(tf_input).astype(np.float32))
                yield input_shapes_tmp

            print(f'{Color.REVERCE}TF-TRT (TensorRT) Float32 convertion started{Color.RESET}', '=' * 40)
            params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP32', maximum_cached_engines=10000)
            converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=saved_model_dir_path, conversion_params=params)
            converter.convert()
            converter.build(input_fn=input_fn)
            converter.save(f'{model_output_dir_path}/tensorrt_saved_model_float32')
            print(f'{Color.GREEN}TF-TRT (TensorRT) convertion complete!{Color.RESET} - {model_output_dir_path}/tensorrt_saved_model_float32')
            print(f'{Color.REVERCE}TF-TRT (TensorRT) Float16 convertion started{Color.RESET}', '=' * 40)
            params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP16', maximum_cached_engines=10000)
            converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=saved_model_dir_path, conversion_params=params)
            converter.convert()
            converter.build(input_fn=input_fn)
            converter.save(f'{model_output_dir_path}/tensorrt_saved_model_float16')
            print(f'{Color.GREEN}TF-TRT (TensorRT) convertion complete!{Color.RESET} - {model_output_dir_path}/tensorrt_saved_model_float16')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()
            print(f'{Color.RED}The binary versions of TensorFlow and TensorRT may not be compatible. Please check the version compatibility of each package.{Color.RESET}')

    # CoreML convert
    if output_coreml:
        try:
            import coremltools as ct
            print(f'{Color.REVERCE}CoreML convertion started{Color.RESET}', '=' * 59)
            mlmodel = ct.convert(saved_model_dir_path, source='tensorflow')
            mlmodel.save(f'{model_output_dir_path}/model_coreml_float32.mlmodel')
            print(f'{Color.GREEN}CoreML convertion complete!{Color.RESET} - {model_output_dir_path}/model_coreml_float32.mlmodel')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model_dir_path', type=str, required=True, help='Input saved_model dir path')
    parser.add_argument('--signature_def', type=str, default='', help='Specifies the signature name to load from saved_model')
    parser.add_argument('--input_shapes', type=str, default='', help='Overwrites an undefined input dimension (None or -1). Specify the input shape in [n,h,w,c] format. For non-4D tensors, specify [a,b,c,d,e], [a,b], etc. A comma-separated list if there are multiple inputs. (e.g.) --input_shapes [1,256,256,3],[1,64,64,3],[1,2,16,16,3]')
    parser.add_argument('--model_output_dir_path', type=str, default='tflite_from_saved_model', help='The output folder path of the converted model file')
    parser.add_argument('--output_no_quant_float32_tflite', type=bool, default=False, help='float32 tflite output switch')
    parser.add_argument('--output_weight_quant_tflite', type=bool, default=False, help='weight quant tflite output switch')
    parser.add_argument('--output_float16_quant_tflite', type=bool, default=False, help='float16 quant tflite output switch')
    parser.add_argument('--output_integer_quant_tflite', type=bool, default=False, help='integer quant tflite output switch')
    parser.add_argument('--output_full_integer_quant_tflite', type=bool, default=False, help='full integer quant tflite output switch')
    parser.add_argument('--output_integer_quant_type', type=str, default='int8', help='Input and output types when doing Integer Quantization (\'int8 (default)\' or \'uint8\')')
    parser.add_argument('--string_formulas_for_normalization', type=str, default='(data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]', help='String formulas for normalization. It is evaluated by Python\'s eval() function. Default: \'(data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]\'')
    parser.add_argument('--calib_ds_type', type=str, default='tfds', help='Types of data sets for calibration. tfds or numpy(Future Implementation)')
    parser.add_argument('--ds_name_for_tfds_for_calibration', type=str, default='coco/2017', help='Dataset name for TensorFlow Datasets for calibration. https://www.tensorflow.org/datasets/catalog/overview')
    parser.add_argument('--split_name_for_tfds_for_calibration', type=str, default='validation', help='Split name for TensorFlow Datasets for calibration. https://www.tensorflow.org/datasets/catalog/overview')
    tfds_dl_default_path = f'{str(Path.home())}/TFDS'
    parser.add_argument('--download_dest_folder_path_for_the_calib_tfds', type=str, default=tfds_dl_default_path, help='Download destination folder path for the calibration dataset. Default: $HOME/TFDS')
    parser.add_argument('--tfds_download_flg', type=bool, default=True, help='True to automatically download datasets from TensorFlow Datasets. True or False')
    parser.add_argument('--output_tfjs', type=bool, default=False, help='tfjs model output switch')
    parser.add_argument('--output_tftrt', type=bool, default=False, help='tftrt model output switch')
    parser.add_argument('--output_coreml', type=bool, default=False, help='coreml model output switch')
    parser.add_argument('--output_edgetpu', type=bool, default=False, help='edgetpu model output switch')

    args = parser.parse_args()
    saved_model_dir_path = args.saved_model_dir_path
    signature_def = args.signature_def
    input_shapes_tmp = args.input_shapes
    input_shapes = []
    if input_shapes_tmp:
        first_digit_reg = r'([0-9 ]+|-1)'
        next_digits_reg = r'(,{})*'.format(first_digit_reg)
        tuple_reg = r'((\({}{}\))|(\[{}{}\]))'.format(first_digit_reg, next_digits_reg, first_digit_reg, next_digits_reg)
        full_reg = r'^{}(\s*,\s*{})*$|^$'.format(tuple_reg, tuple_reg)
        if not re.match(full_reg, input_shapes_tmp):
            print('Input shape "{}" cannot be parsed.', input_shapes_tmp)
        for shape_str in re.findall(r'[(\[]([0-9, -]+)[)\]]', input_shapes_tmp):
            input_shapes.append([int(dim) for dim in shape_str.split(',')])
    model_output_dir_path = args.model_output_dir_path
    output_no_quant_float32_tflite =  args.output_no_quant_float32_tflite
    output_weight_quant_tflite = args.output_weight_quant_tflite
    output_float16_quant_tflite = args.output_float16_quant_tflite
    output_integer_quant_tflite = args.output_integer_quant_tflite
    output_full_integer_quant_tflite = args.output_full_integer_quant_tflite
    output_integer_quant_type = args.output_integer_quant_type.lower()
    string_formulas_for_normalization = args.string_formulas_for_normalization.lower()
    calib_ds_type = args.calib_ds_type.lower()
    ds_name_for_tfds_for_calibration = args.ds_name_for_tfds_for_calibration
    split_name_for_tfds_for_calibration = args.split_name_for_tfds_for_calibration
    download_dest_folder_path_for_the_calib_tfds = args.download_dest_folder_path_for_the_calib_tfds
    tfds_download_flg = args.tfds_download_flg
    output_tfjs = args.output_tfjs
    output_tftrt = args.output_tftrt
    output_coreml = args.output_coreml
    output_edgetpu = args.output_edgetpu

    if not output_no_quant_float32_tflite and \
       not output_weight_quant_tflite and \
       not output_integer_quant_tflite and \
       not output_full_integer_quant_tflite and \
       not output_tfjs and \
       not output_tftrt and \
       not output_coreml and \
       not output_edgetpu:
        print('Set at least one of the output switches (output_*) to true.')
        sys.exit(-1)

    if output_edgetpu:
        output_full_integer_quant_tflite = True

    from pkg_resources import working_set
    package_list = []
    for dist in working_set:
        package_list.append(dist.project_name)

    if output_tfjs:
        if not 'tensorflowjs' in package_list:
            print('\'tensorflowjs\' is not installed. Please run the following command to install \'tensorflowjs\'.')
            print('pip3 install --upgrade tensorflowjs')
            sys.exit(-1)
    if output_tftrt:
        if not 'tensorrt' in package_list:
            print('\'tensorrt\' is not installed. Please check the following website and install \'tensorrt\'.')
            print('https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html')
            sys.exit(-1)
    if output_coreml:
        if not 'coremltools' in package_list:
            print('\'coremltoos\' is not installed. Please run the following command to install \'coremltoos\'.')
            print('pip3 install --upgrade coremltools')
            sys.exit(-1)

    if output_integer_quant_tflite or output_full_integer_quant_tflite:
        if not 'tensorflow-datasets' in package_list:
            print('\'tensorflow-datasets\' is not installed. Please run the following command to install \'tensorflow-datasets\'.')
            print('pip3 install --upgrade tensorflow-datasets')
            sys.exit(-1)

    if output_integer_quant_type == 'int8' or output_integer_quant_type == 'uint8':
        pass
    else:
        print('Only \'int8\' or \'uint8\' can be specified for output_integer_quant_type.')
        sys.exit(-1)

    if calib_ds_type == 'tfds':
        pass
    elif calib_ds_type == 'numpy':
        print('The Numpy mode of the data set for calibration will be implemented in the future.')
        sys.exit(-1)
    else:
        print('Only \'tfds\' or \'numpy\' can be specified for calib_ds_type.')
        sys.exit(-1)

    del package_list
    os.makedirs(model_output_dir_path, exist_ok=True)
    convert(saved_model_dir_path,
            signature_def,
            input_shapes,
            model_output_dir_path,
            output_no_quant_float32_tflite,
            output_weight_quant_tflite,
            output_float16_quant_tflite,
            output_integer_quant_tflite,
            output_full_integer_quant_tflite,
            output_integer_quant_type,
            string_formulas_for_normalization,
            calib_ds_type,
            ds_name_for_tfds_for_calibration,
            split_name_for_tfds_for_calibration,
            download_dest_folder_path_for_the_calib_tfds,
            tfds_download_flg,
            output_tfjs,
            output_tftrt,
            output_coreml,
            output_edgetpu)
    print(f'{Color.REVERCE}All the conversion process is finished!{Color.RESET}', '=' * 45)

if __name__ == "__main__":
    main()
