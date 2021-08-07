#! /usr/bin/env python
'''
tensorflow==2.5.0+

python3 openvino2tensorflow.py \
  --model_path openvino/448x448/FP32/Resnet34_3inputs_448x448_20200609.xml \
  --output_saved_model \
  --output_pb \
  --output_weight_quant_tflite \
  --output_float16_quant_tflit \
  --output_no_quant_float32_tflite
'''
import os
import sys
import argparse
import struct
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as et
from openvino.inference_engine import IECore
import logging
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

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

def convert(model_path,
            model_output_path,
            output_saved_model,
            output_h5,
            output_weight_and_json,
            output_pb,
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
            npy_load_default_path,
            load_dest_file_path_for_the_calib_npy,
            output_tfjs,
            output_tftrt,
            output_coreml,
            output_edgetpu,
            output_onnx,
            onnx_opset,
            output_myriad,
            vpu_number_of_shaves,
            vpu_number_of_cmx_slices,
            replace_swish_and_hardswish,
            optimizing_hardswish_for_edgetpu,
            replace_prelu_and_minmax,
            yolact,
            restricted_resize_image_mode,
            weight_replacement_config,
            debug,
            debug_layer_number):

    print(f'{Color.REVERCE}TensorFlow/Keras model building process starts{Color.RESET}', '=' * 38)

    import subprocess
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    tf.get_logger().setLevel(logging.ERROR)
    import tensorflow_datasets as tfds
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, MaxPool2D, AveragePooling2D, Reshape, Conv2DTranspose, PReLU, Lambda
    from tensorflow.keras.initializers import Constant
    from tensorflow.keras.activations import elu, hard_sigmoid
    from typing import Union
    from tensorflow.python.keras.engine.keras_tensor import KerasTensor
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    if output_coreml:
        import coremltools as ct
    import json
    import pprint

    # for unpacking binary buffer
    format_config = {
        'FP32' : ['f', 4],
        'FP16' : ['e', 2],
        'I64'  : ['q', 8],
        'I32'  : ['i', 4],
        'I16'  : ['h', 2],
        'I8'   : ['b', 1],
        'U8'   : ['B', 1],
        'BOOL' : ['?', 1]
    }

    # vino:    u8,    u16,    u32,    u64,   i8,   i16,   i32,   i64,     f16,     f32,              bf16, boolean
    # tf  : uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float32, float64, bfloat16

    # type conversion table
    cast_type_ov_tf = {
        'u8'  : tf.uint8,
        'u16' : tf.uint16,
        'u32' : tf.uint32,
        'u64' : tf.uint64,
        'i8'  : tf.int8,
        'i16' : tf.int16,
        'i32' : tf.int32,
        'i64' : tf.int64,
        'f16' : tf.float16,
        'f32' : tf.float32,
        'bf16': tf.bfloat16
    }

    # integer type table
    int_type_tf = [
        tf.uint8,
        tf.uint16,
        tf.uint32,
        tf.uint64,
        tf.int8,
        tf.int16,
        tf.int32,
        tf.int64
    ]

    # pad type conversion table
    pad_type_ov_tf = {
        'constant' : 'CONSTANT',
        'reflect'  : 'REFLECT',
        'symmetric': 'SYMMETRIC',
        'edge'     : 'REFLECT'
    }

    # Read IR weight data
    with open(model_path+'.bin', 'rb') as f:
        binWeight = f.read()
    # Parse IR XML file,
    tree = et.parse(model_path+'.xml')
    root = tree.getroot()
    edges = root.find('edges')
    layers = root.find('layers')
    tf_layers_dict = {}
    tf_edges = {}

    tf_inputs = []
    tf_outputs = []
    layer_id_port_dict = {}

    def get_num_of_outputs_per_layer_id(tf_edges):
        output_count_by_layer_id_tmp = {}
        for key in tf_edges.keys():
            key_tmp = key.split(':')[0]
            output_count_by_layer_id_tmp.setdefault(key_tmp, {'count' : 0, 'layer_id:port' : []})
            output_count_by_layer_id_tmp[key_tmp]['count'] += 1
            output_count_by_layer_id_tmp[key_tmp]['layer_id:port'].append(key)
        return output_count_by_layer_id_tmp

    def get_bere_layer_type(before_layer):
        t = type(tf_layers_dict[before_layer])
        if t == np.ndarray:
            # Const
            return 'const'
        else:
            try:
                return tf_layers_dict[before_layer.split(':')[0]].op.type
            except:
                # TopK
                return 'other'

    def get_tf_edges_from(tf_edges, layer_id, edge_index=-1):
        if edge_index == -1:
            # Add, Concat
            layer_list = []
            for edge_index in range(len(tf_edges[layer_id])):
                before_layer_type = get_bere_layer_type(tf_edges[layer_id][edge_index])
                if before_layer_type == 'Split':
                    layer_list.append(tf_edges[layer_id][edge_index])
                elif before_layer_type == 'other':
                    layer_list.append(tf_edges[layer_id][edge_index])
                else:
                    layer_list.append(tf_edges[layer_id][edge_index].split(':')[0])
            return layer_list
        else:
            # Other
            if layer_id in tf_edges:
                before_layer_type = get_bere_layer_type(tf_edges[layer_id][edge_index])
            else:
                for key in tf_edges.keys():
                    if layer_id in key:
                        before_layer_type = get_bere_layer_type(tf_edges[key][edge_index])
                        layer_id = key
                        break
            if before_layer_type == 'Split':
                return tf_edges[layer_id][edge_index]
            elif before_layer_type == 'other':
                return tf_edges[layer_id][edge_index]
            else:
                return tf_edges[layer_id][edge_index].split(':')[0]


    """
    format_version : Format version of weight_replacement_config.
    layer_id : ID of the Const layer whose weight/constant parameter is to be swapped.
               For example, specify "1123" for layer id="1123" for type="Const" in .xml.

               <layer id="1123" name="Decoder/softmax/Reshape_1/Cast_123722_const657_const" type="Const" version="opset1">
                   <data element_type="i64" offset="7632604" shape="4" size="32"/>
                   <output>
                       <port id="1" precision="I64">
                           <dim>4</dim>
                       </port>
                   </output>
               </layer>

    replace_mode : "direct" or "npy"
                   "direct": Specify the values of the Numpy matrix directly in the "values" attribute.
                             Ignores the values recorded in the .bin file and replaces them with the values specified in "values".

                   {
                       "layer_id": "1123",
                       "replace_mode": "direct",
                       "values": [
                           1,
                           2,
                           513,
                           513
                       ]
                   }

                   "npy": Load a Numpy binary file with the matrix output by np.save('xyz', a).
                          The "values" attribute specifies the path to the Numpy binary file.
                   {
                       "layer_id": "1125",
                       "replace_mode": "npy",
                       "values": "weights/xyz.npy"
                   }

    values : Specify the value or the path to the Numpy binary file to replace the weight/constant value recorded in .bin.
             The way to specify is as described in the description of 'replace_mode'.
    """
    def parse_json(jsonfile_path):
        j = json.load(open(jsonfile_path))
        format_version = j['format_version']
        layers = {}
        for v in j['layers']:
            layers[v['layer_id']] = v
        print(f'{Color.GREEN}weight_replacement_config format_version:{Color.RESET} {format_version}')
        print(f'{Color.GREEN}Replace the value of Const for each layer_id with the value below.{Color.RESET}')
        pprint.pprint(layers)
        return layers

    wr_config = None
    if weight_replacement_config:
        wr_config = parse_json(weight_replacement_config)

    # edges
    added_key_list = []
    concat_port_list = {}
    for edge in edges:
        to_layer = edge.attrib['to-layer']
        to_layer_port = edge.attrib['to-port']
        from_layer = edge.attrib['from-layer']
        from_layer_port = edge.attrib['from-port']

        for layer in layers:
            if layer.attrib['id'] == to_layer:
                output_layer_ports = layer.find('output')
                if layer.attrib['type'] != 'Result' and len(output_layer_ports) >= 2:
                    for port in output_layer_ports:
                        tf_edges.setdefault('{}:{}'.format(to_layer, port.attrib['id']), []).append(from_layer)
                    added_key_list.append(to_layer)
                else:
                    tf_edges.setdefault(to_layer, [])
                    if layer.attrib['type'] == 'Concat':
                        concat_port_list.setdefault(to_layer, []).append(f'{from_layer}:{to_layer_port}')

        for layer in layers:
            if layer.attrib['id'] == from_layer:
                output_layer_ports = layer.find('output')
                if len(output_layer_ports) >= 2:
                    tf_edges.setdefault(to_layer, []).append('{}:{}'.format(from_layer, from_layer_port))
                    if to_layer not in added_key_list:
                        added_key_list.append(to_layer)
                else:
                    if to_layer not in added_key_list:
                        tf_edges.setdefault(to_layer, []).append(from_layer)
                        added_key_list.append(to_layer)
                    else:
                        flg = 'not_found'
                        for key in tf_edges.keys():
                            if to_layer in key and from_layer in tf_edges[key]:
                                flg = 'found'
                                break
                        if flg == 'not_found':
                            tf_edges.setdefault(to_layer, []).append(from_layer)
                break

    # The following loop sorts tf_edges in ascending order by port
    # However, only the Concat layer is implemented.
    for to_layer, from_layer_ports in concat_port_list.items():
        temp_sorted_tf_edge = []
        # from_layer_ports = [from_layer_id:port, from_layer_id:port, from_layer_id:port, ...]
        ports = [p.split(':')[1] for p in from_layer_ports]
        for idx, port in enumerate(ports):
            temp_sorted_tf_edge.append(tf_edges[to_layer][ports.index(str(idx))])
        tf_edges[to_layer] = temp_sorted_tf_edge

    del added_key_list
    del concat_port_list

    layer_id_port_dict = get_num_of_outputs_per_layer_id(tf_edges)

    process_interruption_by_non_max_suppression = False

    # layers
    for idx, layer in enumerate(layers):

        layer_id = layer.attrib['id']
        layer_name = layer.attrib['name'].replace('.', '_').replace('/', '_')
        data = layer.find('data')

        try:
            ### Parameter
            if layer.attrib['type'] == 'Parameter':
                if not data is None and 'shape' in data.attrib:
                    shape_str  = data.attrib['shape'].split(',')
                    shape = [int(s) for s in shape_str]
                    if len(shape) == 4:
                        tf_layers_dict[layer_id] = Input(shape=(shape[2], shape[3], shape[1]), batch_size=shape[0], name=layer_name)
                    else:
                        tf_layers_dict[layer_id] = Input(shape=[inp for inp in shape[1:]], batch_size=shape[0], name=layer_name)
                    tf_inputs.append(tf_layers_dict[layer_id])

            ### Const
            elif layer.attrib['type'] == 'Const':
                if not data is None:
                    if 'offset' in data.attrib and 'size' in data.attrib:
                        offset = int(data.attrib['offset'])
                        size   = int(data.attrib['size'])
                        shape_str = '1' if data.attrib['shape'] == '' else data.attrib['shape'].split(',')
                        shape = [int(s) for s in shape_str]
                        blobBin = binWeight[offset:offset+size]
                        prec = layer.find('output').find('port').attrib['precision']
                        formatstring = '<' + format_config[prec][0] * (len(blobBin)//format_config[prec][1])
                        decodedwgt = np.array(list(struct.unpack(formatstring, blobBin))).reshape(shape)

                        if not wr_config or layer_id not in wr_config:
                            if type(decodedwgt) == np.ndarray and decodedwgt.dtype == np.float64:
                                tf_layers_dict[layer_id] = decodedwgt.astype(np.float32)
                            else:
                                tf_layers_dict[layer_id] = decodedwgt
                        else:
                            if layer_id in wr_config:
                                if wr_config[layer_id]['replace_mode'] == 'direct':
                                    try:
                                        tf_layers_dict[layer_id] = np.array(wr_config[layer_id]['values'])
                                    except:
                                        tf_layers_dict[layer_id] = wr_config[layer_id]['values']
                                elif wr_config[layer_id]['replace_mode'] == 'npy':
                                    tf_layers_dict[layer_id] = np.load(wr_config[layer_id]['values'])
                                else:
                                    mode_str = wr_config[layer_id]['replace_mode']
                                    print(f'replace_mode = {mode_str} is not supported. Please review the weight_replacement_config json.')
                                    sys.exit(-1)
                            else:
                                if type(decodedwgt) == np.ndarray and decodedwgt.dtype == np.float64:
                                    tf_layers_dict[layer_id] = decodedwgt.astype(np.float32)
                                else:
                                    tf_layers_dict[layer_id] = decodedwgt

            ### Convolution
            elif layer.attrib['type'] == 'Convolution':
                # port0 = [int(sdim.text) for sdim in layer.find('input')[0]]
                port1 = [int(sdim.text) for sdim in layer.find('input')[1]]
                filters = int(port1[0])
                kernel_size = [int(port1[2]), int(port1[3])]
                strides = [int(s) for s in data.attrib['strides'].split(',')]
                pads_begin = 0
                pads_end = 0
                if not data is None and 'pads_begin' in data.attrib:
                    pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
                if not data is None and 'pads_end' in data.attrib:
                    pads_end = sum([int(s) for s in data.attrib['pads_end'].split(',')])
                padding = ''
                if (pads_begin + pads_end) == 0:
                    if 'auto_pad' in data.attrib:
                        if data.attrib['auto_pad'] == 'same_upper' or data.attrib['auto_pad'] == 'same_lower':
                            padding = 'same'
                        else:
                            padding = 'valid'
                    else:
                        padding = 'valid'
                else:
                    padding = 'same'

                temp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                orig = None
                if pads_begin > 0:
                    padding = 'valid'
                    # begin 0 = top
                    # begin 1 = left
                    # end 0 = bottom
                    # end 1 = right
                    begin = [int(data.attrib['pads_begin'].split(',')[0]), int(data.attrib['pads_end'].split(',')[0])]
                    end   = [int(data.attrib['pads_begin'].split(',')[1]), int(data.attrib['pads_end'].split(',')[1])]
                    orig = tf.keras.layers.ZeroPadding2D([begin, end])(temp)
                else:
                    if temp.shape[0] == 1 and temp.shape[2] == 1 and temp.shape[3] == 1:
                        orig = tf.transpose(temp, perm=(0,2,3,1))
                    else:
                        orig = temp

                dilations = [int(s) for s in data.attrib['dilations'].split(',')]
                try:
                    tf_layers_dict[layer_id] = Conv2D(
                        filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        dilation_rate=dilations,
                        use_bias=False,
                        kernel_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(2,3,1,0)))(orig)
                except:
                    try:
                        tf_layers_dict[layer_id] = Conv2D(
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            dilation_rate=dilations,
                            use_bias=False,
                            kernel_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].numpy().transpose(2,3,1,0)))(orig)
                    except:
                        # Weights from OP that are not fixed values
                        tf_layers_dict[layer_id] = tf.nn.conv2d(
                            input=orig,
                            filters=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                            strides=strides,
                            padding=padding.upper(),
                            dilations=dilations
                        )

            ### Add
            elif layer.attrib['type'] == 'Add':
                # 'Fused_Add_' == BiasAdd
                if len(tf_edges[layer_id]) == 2 and type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) == np.ndarray:
                    try:
                        # Biasadd or Add
                        edge_id0 = get_tf_edges_from(tf_edges, layer_id, 0)
                        edge_id1 = get_tf_edges_from(tf_edges, layer_id, 1)
                        if tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape[-1] == tf_layers_dict[edge_id1].flatten().shape[0]:
                            tf_layers_dict[layer_id] = tf.math.add(tf_layers_dict[edge_id0], tf_layers_dict[edge_id1].flatten())
                        else:
                            tf_layers_dict[layer_id] = tf.math.add(tf_layers_dict[edge_id0], tf_layers_dict[edge_id1])
                    except:
                        # Add
                        edge_id0 = get_tf_edges_from(tf_edges, layer_id, 0)
                        edge_id1 = get_tf_edges_from(tf_edges, layer_id, 1)
                        try:
                            tf_layers_dict[layer_id] = tf.math.add(tf_layers_dict[edge_id0], tf_layers_dict[edge_id1])
                        except:
                            tf_layers_dict[layer_id] = tf.math.add(tf_layers_dict[edge_id0], tf_layers_dict[edge_id1].transpose(0,2,3,1))
                else:
                    # Add
                    if len(get_tf_edges_from(tf_edges, layer_id)) == 2:
                        try:
                            tmp_layers = [tf_layers_dict[from_layer_id].transpose(0,2,3,1).astype(np.float32) if type(tf_layers_dict[from_layer_id]) == np.ndarray else tf_layers_dict[from_layer_id] for from_layer_id in get_tf_edges_from(tf_edges, layer_id)]
                            tf_layers_dict[layer_id] = tf.math.add(tmp_layers[0], tmp_layers[1])
                        except:
                            try:
                                tmp_layers = [tf_layers_dict[from_layer_id].transpose(0,2,3,1) if type(tf_layers_dict[from_layer_id]) == np.ndarray else tf_layers_dict[from_layer_id] for from_layer_id in get_tf_edges_from(tf_edges, layer_id)]
                                tf_layers_dict[layer_id] = tf.math.add(tmp_layers[0], tmp_layers[1])
                            except:
                                tmp_layers = [tf_layers_dict[from_layer_id] if type(tf_layers_dict[from_layer_id]) == np.ndarray else tf_layers_dict[from_layer_id] for from_layer_id in get_tf_edges_from(tf_edges, layer_id)]
                                tf_layers_dict[layer_id] = tf.math.add(tmp_layers[0], tmp_layers[1])
                    else:
                        tf_layers_dict[layer_id] = tf.math.add_n(
                            [tf_layers_dict[from_layer_id].transpose(0,2,3,1).astype(np.float32) if type(tf_layers_dict[from_layer_id]) == np.ndarray else tf_layers_dict[from_layer_id] for from_layer_id in get_tf_edges_from(tf_edges, layer_id)]
                        )

            ### ReLU
            elif layer.attrib['type'] == 'ReLU':
                tf_layers_dict[layer_id] = tf.nn.relu(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### PReLU
            elif layer.attrib['type'] == 'PReLU':
                input_len = len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)
                alpha_len = len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].shape)

                shared_axes = []
                if alpha_len < 4:
                    shared_axes = [val + 1 for val in range(input_len - 1)]
                else:
                    shared_axes = None

                if alpha_len == 4:
                    if replace_prelu_and_minmax:
                        tf_layers_dict[layer_id] = \
                            tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + \
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(0,2,3,1) * tf.minimum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                    else:
                        tf_layers_dict[layer_id] = PReLU(
                            alpha_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(0,2,3,1)),
                            shared_axes=shared_axes)(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )
                else:
                    if replace_prelu_and_minmax:
                        tf_layers_dict[layer_id] = \
                            tf.maximum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) + \
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)] * tf.minimum(0.0, tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                    else:
                        tf_layers_dict[layer_id] = PReLU(
                            alpha_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]),
                            shared_axes=shared_axes)(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                        )

            ### Clamp
            elif layer.attrib['type'] == 'Clamp':
                cmin = None
                cmax = None
                if tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].dtype == np.int64:
                    cmin = np.asarray(data.attrib['min']).astype(np.int64)
                    cmax = np.asarray(data.attrib['max']).astype(np.int64)
                elif tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].dtype == np.int32:
                    cmin = np.asarray(data.attrib['min']).astype(np.int32)
                    cmax = np.asarray(data.attrib['max']).astype(np.int32)
                elif tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].dtype == np.float32:
                    cmin = np.asarray(data.attrib['min']).astype(np.float32)
                    cmax = np.asarray(data.attrib['max']).astype(np.float32)
                else:
                    cmin = np.asarray(data.attrib['min']).astype(np.float32)
                    cmax = np.asarray(data.attrib['max']).astype(np.float32)

                if cmin == 0.0 and cmax == 6.0:
                    # ReLU6
                    tf_layers_dict[layer_id] = tf.nn.relu6(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                else:
                    # Other
                    tf_layers_dict[layer_id] = tf.clip_by_value(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        clip_value_min=cmin,
                        clip_value_max=cmax
                    )

            ### Tan
            elif layer.attrib['type'] == 'Tan':
                tf_layers_dict[layer_id] = tf.math.tan(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Tanh
            elif layer.attrib['type'] == 'Tanh':
                tf_layers_dict[layer_id] = tf.math.tanh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Elu
            elif layer.attrib['type'] == 'Elu':
                alpha = float(data.attrib['alpha'])
                tf_layers_dict[layer_id] = elu(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], alpha=alpha)

            ### HardSigmoid
            elif layer.attrib['type'] == 'HardSigmoid':
                tf_layers_dict[layer_id] = hard_sigmoid(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Sigmoid
            elif layer.attrib['type'] == 'Sigmoid':
                tf_layers_dict[layer_id] = tf.math.sigmoid(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Swish
            elif layer.attrib['type'] == 'Swish':
                if replace_swish_and_hardswish:
                    # Hard-Swish
                    if not optimizing_hardswish_for_edgetpu:
                        tf_layers_dict[layer_id] = \
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] * tf.nn.relu6(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] + 3) * 0.16666667
                    else:
                        tf_layers_dict[layer_id] = \
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] * tf.nn.relu6(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] + 3) * 0.16666666
                else:
                    # Swish
                    tf_layers_dict[layer_id] = tf.nn.swish(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### SoftPlus
            elif layer.attrib['type'] == 'SoftPlus':
                tf_layers_dict[layer_id] = tf.math.softplus(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### MaxPool
            elif layer.attrib['type'] == 'MaxPool':
                outport_size = sum([int(sdim.text) for sdim in layer.find('output')[0]])
                kernel_size =  [int(s) for s in data.attrib['kernel'].split(',')]
                strides = [int(s) for s in data.attrib['strides'].split(',')]
                pads_begin = 0
                pads_end = 0
                if not data is None and 'pads_begin' in data.attrib:
                    pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
                if not data is None and 'pads_end' in data.attrib:
                    pads_end = sum([int(s) for s in data.attrib['pads_end'].split(',')])
                padding = ''
                if (pads_begin + pads_end) == 0:
                    if 'auto_pad' in data.attrib:
                        if data.attrib['auto_pad'] == 'same_upper' or data.attrib['auto_pad'] == 'same_lower':
                            padding = 'SAME'
                        else:
                            padding = 'VALID'
                    else:
                        padding = 'VALID'
                else:
                    padding = 'SAME'

                tf_layers_dict[layer_id] = tf.nn.max_pool(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    ksize=kernel_size,
                    strides=strides,
                    padding=padding
                )
                new_layer_outport_size = sum([sdim for sdim in tf_layers_dict[layer_id].shape])
                if outport_size != new_layer_outport_size:
                    # Caffe -> TF
                    tf_layers_dict[layer_id] = tf.nn.max_pool(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        ksize=kernel_size,
                        strides=strides,
                        padding='SAME'
                    )

            ### AvgPool
            elif layer.attrib['type'] == 'AvgPool':
                kernel_size =  [int(s) for s in data.attrib['kernel'].split(',')]
                strides = [int(s) for s in data.attrib['strides'].split(',')]
                # exclude_pad = data.attrib['exclude-pad']
                pads_begin = 0
                pads_end = 0
                if not data is None and 'pads_begin' in data.attrib:
                    pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
                if not data is None and 'pads_end' in data.attrib:
                    pads_end = sum([int(s) for s in data.attrib['pads_end'].split(',')])
                padding = ''
                if (pads_begin + pads_end) == 0:
                    if 'auto_pad' in data.attrib:
                        if data.attrib['auto_pad'] == 'same_upper' or data.attrib['auto_pad'] == 'same_lower':
                            padding = 'same'
                        else:
                            padding = 'valid'
                    else:
                        padding = 'valid'
                else:
                    padding = 'same'
                if not data is None and 'rounding_type' in data.attrib and data.attrib['rounding_type'] == 'ceil':
                    padding = 'same'
                tf_layers_dict[layer_id] = AveragePooling2D(
                    pool_size=kernel_size,
                    strides=strides,
                    padding=padding
                )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### GroupConvolution
            elif layer.attrib['type'] == 'GroupConvolution':
                port0 = [int(sdim.text) for sdim in layer.find('input')[0]]
                port1 = [int(sdim.text) for sdim in layer.find('input')[1]]
                depth_multiplier = 1
                kernel_size = [int(port1[3]), int(port1[4])]
                strides = [int(s) for s in data.attrib['strides'].split(',')]
                pads_begin = 0
                pads_end = 0
                if not data is None and 'pads_begin' in data.attrib:
                    pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
                if not data is None and 'pads_end' in data.attrib:
                    pads_end = sum([int(s) for s in data.attrib['pads_end'].split(',')])
                padding = ''
                if (pads_begin + pads_end) == 0:
                    if 'auto_pad' in data.attrib:
                        if data.attrib['auto_pad'] == 'same_upper' or data.attrib['auto_pad'] == 'same_lower':
                            padding = 'same'
                        else:
                            padding = 'valid'
                    else:
                        padding = 'valid'
                else:
                    padding = 'same'

                temp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                orig = None
                if pads_begin > 0:
                    padding = 'valid'
                    # begin 0 = top
                    # begin 1 = left
                    # end 0 = bottom
                    # end 1 = right
                    begin = [int(data.attrib['pads_begin'].split(',')[0]), int(data.attrib['pads_end'].split(',')[0])]
                    end   = [int(data.attrib['pads_begin'].split(',')[1]), int(data.attrib['pads_end'].split(',')[1])]
                    orig = tf.keras.layers.ZeroPadding2D([begin, end])(temp)
                else:
                    orig = temp

                dilations = [int(s) for s in data.attrib['dilations'].split(',')]

                if int(port1[1]) > 1:
                    # Conv2D with groups
                    filters = int(port0[1])
                    groups = int(port1[0])

                    convs = []
                    kernel = None
                    if len(port1) == 5:
                        try:
                            kernel = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(3,4,2,1,0)
                        except:
                            kernel = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].numpy().transpose(3,4,2,1,0)

                        for i in range(groups):
                            convs.append(Conv2D(filters=filters // groups,
                                                kernel_size=kernel_size,
                                                strides=strides,
                                                padding=padding,
                                                dilation_rate=dilations,
                                                use_bias=False,
                                                kernel_initializer=Constant(kernel[:,:,:,:,i])))
                    else:
                        try:
                            kernel = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(2,3,1,0)
                        except:
                            kernel = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].numpy().transpose(2,3,1,0)
                        for i in range(groups):
                            convs.append(Conv2D(filters=filters // groups,
                                                kernel_size=kernel_size,
                                                strides=strides,
                                                padding=padding,
                                                dilation_rate=dilations,
                                                use_bias=False,
                                                kernel_initializer=Constant(kernel[:,:,:,i])))

                    x_splits = tf.split(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], groups, -1)
                    x_outputs = [conv(x_split) for x_split, conv in zip(x_splits, convs)]
                    tf_layers_dict[layer_id] = tf.concat(x_outputs, -1)

                else:
                    # DepthwiseConv2D
                    try:
                        tf_layers_dict[layer_id] = DepthwiseConv2D(kernel_size=kernel_size,
                                                                strides=strides,
                                                                padding=padding,
                                                                depth_multiplier=depth_multiplier,
                                                                dilation_rate=dilations,
                                                                use_bias=False,
                                                                depthwise_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(3,4,1,2,0))
                                                            )(orig)
                    except:
                        tf_layers_dict[layer_id] = DepthwiseConv2D(kernel_size=kernel_size,
                                                                strides=strides,
                                                                padding=padding,
                                                                depth_multiplier=depth_multiplier,
                                                                dilation_rate=dilations,
                                                                use_bias=False,
                                                                depthwise_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].numpy().transpose(3,4,1,2,0))
                                                            )(orig)

            ### ConvolutionBackpropData
            elif layer.attrib['type'] == 'ConvolutionBackpropData':
                # port0 = [int(sdim.text) for sdim in layer.find('input')[0]]
                port1 = [int(sdim.text) for sdim in layer.find('input')[1]]
                port2 = [int(sdim.text) for sdim in layer.find('output')[0]]
                filters = int(port2[1])
                kernel_size = [int(port1[2]), int(port1[3])]
                strides = [int(s) for s in data.attrib['strides'].split(',')]
                pads_begin = 0
                pads_end = 0
                if not data is None and 'pads_begin' in data.attrib:
                    pads_begin = sum([int(s) for s in data.attrib['pads_begin'].split(',')])
                if not data is None and 'pads_end' in data.attrib:
                    pads_end = sum([int(s) for s in data.attrib['pads_end'].split(',')])
                padding = ''
                if (pads_begin + pads_end) == 0:
                    if 'auto_pad' in data.attrib:
                        if data.attrib['auto_pad'] == 'same_upper' or data.attrib['auto_pad'] == 'same_lower':
                            padding = 'same'
                        else:
                            padding = 'valid'
                    else:
                        padding = 'valid'
                else:
                    padding = 'same'
                dilations = [int(s) for s in data.attrib['dilations'].split(',')]
                tf_layers_dict[layer_id] = Conv2DTranspose(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    dilation_rate=dilations,
                    use_bias=False,
                    kernel_initializer=Constant(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(2,3,1,0))
                )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Concat
            elif layer.attrib['type'] == 'Concat':
                axis = -1
                if 'axis' in data.attrib:
                    axis = int(data.attrib['axis'])
                if axis == 1 and len(np.asarray(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)) == 4:
                    axis = -1

                length_list = []
                for from_layer_id in get_tf_edges_from(tf_edges, layer_id):
                    length_list.append(len(tf_layers_dict[from_layer_id].shape))
                axis_zero_count = length_list.count(0)
                sum_of_axis = sum(length_list)
                if axis_zero_count > 0 and sum_of_axis > 0:
                    tensor_list = []
                    for length, from_layer_id in zip(length_list, get_tf_edges_from(tf_edges, layer_id)):
                        if length == 0:
                            tensor_list.append([tf_layers_dict[from_layer_id]])
                        else:
                            tensor_list.append(tf_layers_dict[from_layer_id])
                    tf_layers_dict[layer_id] = tf.concat(tensor_list, axis=axis)
                else:
                    temp = get_tf_edges_from(tf_edges, layer_id)
                    tf_layers_dict[layer_id] = tf.concat(
                        [tf_layers_dict[from_layer_id] for from_layer_id in get_tf_edges_from(tf_edges, layer_id)],
                        axis=axis
                    )

            ### Multiply
            elif layer.attrib['type'] == 'Multiply':
                if len(tf_edges[layer_id]) == 2 and (type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) == np.ndarray):
                    if tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].ndim == 4:
                        # 4D - NCHW->NHWC
                        tf_layers_dict[layer_id] = tf.math.multiply(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(0,2,3,1).astype(np.float32)
                        )
                    else:
                        # unknown
                        try:
                            x_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].type_spec.shape
                            y_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].shape
                            if x_shape == y_shape:
                                tf_layers_dict[layer_id] = tf.math.multiply(
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                                )
                            else:
                                try:
                                    tf_layers_dict[layer_id] = tf.math.multiply(
                                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].reshape(x_shape)
                                    )
                                except:
                                    try:
                                        tf_layers_dict[layer_id] = tf.math.multiply(
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].transpose(0,2,1)
                                        )
                                    except:
                                        tf_layers_dict[layer_id] = tf.math.multiply(
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                                        )
                        except:
                            tf_layers_dict[layer_id] = tf.math.multiply(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                            )
                elif len(tf_edges[layer_id]) == 2 and (type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]) == np.ndarray):
                    if tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].ndim == 4:
                        # 4D - NCHW->NHWC
                        tf_layers_dict[layer_id] = tf.math.multiply(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].transpose(0,2,3,1).astype(np.float32),
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                        )
                    else:
                        # unknown
                        try:
                            x_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape
                            y_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].type_spec.shape
                            if x_shape == y_shape:
                                tf_layers_dict[layer_id] = tf.math.multiply(
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                                )
                            else:
                                try:
                                    tf_layers_dict[layer_id] = tf.math.multiply(
                                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].reshape(y_shape),
                                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                                    )
                                except:
                                    try:
                                        tf_layers_dict[layer_id] = tf.math.multiply(
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].transpose(0,2,1),
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                                        )
                                    except:
                                        tf_layers_dict[layer_id] = tf.math.multiply(
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                                        )
                        except:
                            tf_layers_dict[layer_id] = tf.math.multiply(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                            )
                else:
                    # unknown
                    tf_layers_dict[layer_id] = tf.math.multiply(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                    )

            ### Interpolate
            elif layer.attrib['type'] == 'Interpolate':
                mode = data.attrib['mode']
                if mode == 'linear_onnx':
                    mode = 'linear'
                antialias = False
                try:
                    antialias = False if int(data.attrib['antialias']) == 0 else True
                except:
                    antialias = False if data.attrib['antialias'].upper() == 'FALSE' else True
                out_port0 = [int(sdim.text) for sdim in layer.find('output')[0]]
                out_height = int(out_port0[2])
                out_width  = int(out_port0[3])

                input_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape
                input_shape_height = input_shape[1]
                input_shape_width  = input_shape[2]
                upsampling_factor_height = out_height // input_shape_height
                upsampling_factor_width  = out_width // input_shape_width

                def upsampling2d_bilinear(x, upsampling_factor_height, upsampling_factor_width):
                    h = x.shape[1] * upsampling_factor_height
                    w = x.shape[2] * upsampling_factor_width
                    if output_edgetpu:
                        print(f'{Color.RED}WARNING:{Color.RESET} The weights after Upsampling (tf.compat.v1.image.resize_bilinear) are shifted to the upper left. If you do not need to generate EdgeTPU models, set --output_edgetpu False and run again. OP: {x.name}')
                        return tf.compat.v1.image.resize_bilinear(x, (h, w))#, align_corners=True, half_pixel_centers=True)
                    else:
                        return tf.image.resize(x, [h, w], method='bilinear')

                def upsampling2d_nearest(x, upsampling_factor_height, upsampling_factor_width):
                    h = x.shape[1] * upsampling_factor_height
                    w = x.shape[2] * upsampling_factor_width
                    if output_edgetpu:
                        print(f'{Color.RED}WARNING:{Color.RESET} The weights after Upsampling (tf.compat.v1.image.resize_nearest_neighbor) are shifted to the upper left. If you do not need to generate EdgeTPU models, set --output_edgetpu False and run again. OP: {x.name}')
                        return tf.compat.v1.image.resize_nearest_neighbor(x, (h, w))#, align_corners=True, half_pixel_centers=True)
                    else:
                        return tf.image.resize(x, [h, w], method='nearest')

                if not restricted_resize_image_mode:

                    if (upsampling_factor_height * input_shape_height) == out_height and \
                        (upsampling_factor_width * input_shape_width) == out_width and \
                            upsampling_factor_height >= 1.0 and \
                                upsampling_factor_width >= 1.0:
                        # Upsampling
                        if mode == 'linear':
                            tf_layers_dict[layer_id] = Lambda(upsampling2d_bilinear,
                                                                arguments={'upsampling_factor_height': upsampling_factor_height,
                                                                        'upsampling_factor_width': upsampling_factor_width}
                                                            )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        elif mode == 'nearest':
                            tf_layers_dict[layer_id] = Lambda(upsampling2d_nearest,
                                                                arguments={'upsampling_factor_height': upsampling_factor_height,
                                                                        'upsampling_factor_width': upsampling_factor_width}
                                                            )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                        else:
                            print(f'The Interpolate - {mode} is not yet implemented.')
                            sys.exit(-1)
                    else:
                        # Others
                        if yolact:
                            if mode == 'linear':
                                x = Lambda(upsampling2d_bilinear,
                                            arguments={'upsampling_factor_height': 2,
                                                    'upsampling_factor_width':  2}
                                        )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                tf_layers_dict[layer_id] = tf.slice(x, [0, 1, 1, 0], [-1, -1, -1, -1])
                            elif mode == 'nearest':
                                x = Lambda(upsampling2d_nearest,
                                        arguments={'upsampling_factor_height': 2,
                                                    'upsampling_factor_width':  2}
                                        )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                                tf_layers_dict[layer_id] = tf.slice(x, [0, 1, 1, 0], [-1, -1, -1, -1])
                            else:
                                print(f'The Interpolate - {mode} is not yet implemented.')
                                sys.exit(-1)
                        else:
                            if mode == 'linear':
                                if output_edgetpu:
                                    tf_layers_dict[layer_id] = tf.compat.v1.image.resize(
                                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                        [out_height, out_width],
                                        method='bilinear'
                                    )
                                else:
                                    tf_layers_dict[layer_id] = tf.image.resize(
                                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                        [out_height, out_width],
                                        method='bilinear'
                                    )
                            elif mode == 'nearest':
                                if output_edgetpu:
                                    tf_layers_dict[layer_id] = tf.compat.v1.image.resize(
                                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                        [out_height, out_width],
                                        method='nearest'
                                    )
                                else:
                                    tf_layers_dict[layer_id] = tf.image.resize(
                                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                        [out_height, out_width],
                                        method='nearest'
                                    )
                            else:
                                print(f'The Interpolate - {mode} is not yet implemented.')
                                sys.exit(-1)
                else:
                    if mode == 'linear':
                        tf_layers_dict[layer_id] = tf.image.resize(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            [out_height, out_width],
                            method='bilinear'
                        )
                    elif mode == 'nearest':
                        tf_layers_dict[layer_id] = tf.image.resize(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            [out_height, out_width],
                            method='nearest'
                        )
                    else:
                        print(f'The Interpolate - {mode} is not yet implemented.')
                        sys.exit(-1)

            ### ShapeOf
            elif layer.attrib['type'] == 'ShapeOf':
                try:
                    tf_layers_dict[layer_id] = tf.constant(
                        np.asarray(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].type_spec.shape),
                        dtype=tf.int64
                    )
                except:
                    tf_layers_dict[layer_id] = tf.shape(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        out_type=tf.int64
                    )

            ### Convert
            elif layer.attrib['type'] == 'Convert':
                # vino:    u8,    u16,    u32,    u64,   i8,   i16,   i32,   i64,     f16,     f32,              bf16, boolean
                # tf  : uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float32, float64, bfloat16
                destination_type = data.attrib['destination_type']
                tf_layers_dict[layer_id] = tf.cast(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    cast_type_ov_tf[destination_type]
                )

            ### StridedSlice - TODO
            elif layer.attrib['type'] == 'StridedSlice':
                begin_mask       = data.attrib['begin_mask']
                end_mask         = data.attrib['end_mask']
                ellipsis_mask    = data.attrib['ellipsis_mask']
                new_axis_mask    = data.attrib['new_axis_mask']
                shrink_axis_mask = data.attrib['shrink_axis_mask']

                # begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask
                begin_mask       = np.asarray([int(val) for val in begin_mask.split(',')])
                end_mask         = np.asarray([int(val) for val in end_mask.split(',')])
                ellipsis_mask    = np.asarray([int(val) for val in ellipsis_mask.split(',')])
                new_axis_mask    = np.asarray([int(val) for val in new_axis_mask.split(',')])
                shrink_axis_mask = np.asarray([int(val) for val in shrink_axis_mask.split(',')])

                if type(begin_mask) == np.ndarray and len(begin_mask) == 4:
                    begin_mask[0], begin_mask[1], begin_mask[2], begin_mask[3] = \
                        begin_mask[0], begin_mask[2], begin_mask[3], begin_mask[1]
                if np.sum(begin_mask) == len(begin_mask):
                    begin_mask = -1
                else:
                    begin_mask = np.argmin(begin_mask)

                if type(end_mask) == np.ndarray and len(end_mask) == 4:
                    end_mask[0], end_mask[1], end_mask[2], end_mask[3] = \
                        end_mask[0], end_mask[2], end_mask[3], end_mask[1]
                if np.sum(end_mask) == len(end_mask):
                    end_mask = -1
                else:
                    end_mask = np.argmin(end_mask)

                if type(ellipsis_mask) == np.ndarray and len(ellipsis_mask) == 4:
                    ellipsis_mask[0], ellipsis_mask[1], ellipsis_mask[2], ellipsis_mask[3] = \
                        ellipsis_mask[0], ellipsis_mask[2], ellipsis_mask[3], ellipsis_mask[1]
                ellipsis_mask = np.argmin(ellipsis_mask)

                if type(new_axis_mask) == np.ndarray and len(new_axis_mask) == 4:
                    new_axis_mask[0], new_axis_mask[1], new_axis_mask[2], new_axis_mask[3] = \
                        new_axis_mask[0], new_axis_mask[2], new_axis_mask[3], new_axis_mask[1]
                new_axis_mask = np.argmin(new_axis_mask)


                if type(shrink_axis_mask) == np.ndarray and len(shrink_axis_mask) == 4:
                    shrink_axis_mask[0], shrink_axis_mask[1], shrink_axis_mask[2], shrink_axis_mask[3] = \
                        shrink_axis_mask[0], shrink_axis_mask[2], shrink_axis_mask[3], shrink_axis_mask[1]
                shrink_axis_mask = np.argmin(shrink_axis_mask)

                # begin, end, strides
                begin   = np.asarray([int(val) for val in tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]])
                end     = np.asarray([int(val) for val in tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]])
                strides = np.asarray([int(val) for val in tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 3)]])

                if len(begin) == 4:
                    begin[0], begin[1], begin[2], begin[3] = begin[0], begin[2], begin[3], begin[1]

                if len(end) == 4:
                    end[0], end[1], end[2], end[3] = end[0], end[2], end[3], end[1]

                for idx, (b, e) in enumerate(zip(begin, end)):
                    if b == 0 and b == e:
                        begin[idx] = 0
                        end[idx] = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape[idx]

                if len(strides) == 4:
                    strides[0], strides[1], strides[2], strides[3] = strides[0], strides[2], strides[3], strides[1]

                tf_layers_dict[layer_id] = tf.strided_slice(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    begin=begin,
                    end=end,
                    strides=strides,
                    begin_mask=begin_mask,
                    end_mask=end_mask,
                    ellipsis_mask=ellipsis_mask,
                    new_axis_mask=new_axis_mask,
                    shrink_axis_mask=shrink_axis_mask
                )

            ### Pad
            elif layer.attrib['type'] == 'Pad':
                pad_mode = pad_type_ov_tf[data.attrib['pad_mode']]
                pads_begin = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)] # [0,0,1,1]
                pads_end   = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)] # [0,0,1,1]

                if pad_mode != 'CONSTANT':
                    if (pads_end[0] == 0 and pads_begin[0] == 0):
                        pad_b_bottom = pad_b_top = 0
                    else:
                        pad_b_top = pads_begin[0]
                        pad_b_bottom = pads_end[0]

                    if (pads_end[1] == 0 and pads_begin[1] == 0):
                        pad_c_bottom = pad_c_top = 0
                    else:
                        pad_c_top = pads_begin[1]
                        pad_c_bottom = pads_end[1]

                    if (pads_end[2] == 0 and pads_begin[2] == 0):
                        pad_bottom = pad_top = 0
                    else:
                        pad_top = pads_begin[2]
                        pad_bottom = pads_end[2]

                    if (pads_end[3] == 0 and pads_begin[3] == 0):
                        pad_right = pad_left = 0
                    else:
                        pad_left = pads_begin[3]
                        pad_right = pads_end[3]
                    paddings = [[pad_b_top, pad_b_bottom], [pad_top, pad_bottom], [pad_left, pad_right], [pad_c_top, pad_c_bottom]]

                else:
                    paddings = [pads_begin, pads_end]

                pad_value  = np.float32(0.0)
                if 'pad_value' in data.attrib:
                    pad_value = np.float32(data.attrib['pad_value'])

                try:
                    tf_layers_dict[layer_id] = tf.pad(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        paddings,
                        mode=pad_mode,
                        constant_values=pad_value
                    )
                except:
                    # workaround
                    tf_layers_dict[layer_id] = tf.identity(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                    )

            ### TopK
            elif layer.attrib['type'] == 'TopK':
                # axis = int(data.attrib['axis'])
                # index_element_type = data.attrib['index_element_type']
                # mode = data.attrib['mode']
                # sort = data.attrib['sort']
                layer_id_values  = layer_id_port_dict[layer_id]['layer_id:port'][0]
                layer_id_indices = layer_id_port_dict[layer_id]['layer_id:port'][1]
                try:
                    tf_layers_dict[layer_id_values], tf_layers_dict[layer_id_indices] = \
                        tf.math.top_k(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            k=int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]),
                            sorted=True
                        )
                except:
                    tf_layers_dict[layer_id_values], tf_layers_dict[layer_id_indices] = \
                        tf.math.top_k(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            k=int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)][0]),
                            sorted=True
                        )

            ### Transpose
            elif layer.attrib['type'] == 'Transpose':
                input_shape_len = len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)
                temp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                perm = []
                if input_shape_len == 4:
                    if type(temp) == np.ndarray:
                        if np.all(temp == [0,3,1,2]):
                            for idx, dim in enumerate(temp):
                                perm.append(idx)
                        elif np.all(temp == [1,0,2,3]):
                            perm = [3,1,2,0]
                        else:
                            for idx, dim in enumerate(temp):
                                if dim == 0:
                                    perm.append(0)
                                elif dim == 1:
                                    perm.append(input_shape_len - 1)
                                else:
                                    perm.append(dim - 1)
                    else:
                        # TODO
                        shape = tf.shape(temp)
                        for idx, dim in enumerate(shape):
                            if dim == 0:
                                perm.append(0)
                            elif dim == 1:
                                perm.append(input_shape_len - 1)
                            else:
                                perm.append(dim - 1)
                elif input_shape_len == 5:
                    if np.all(temp == [0,2,1,3,4]):
                        perm.append(0)
                        perm.append(1)
                        perm.append(2)
                        perm.append(4)
                        perm.append(3)
                    else:
                        # TODO
                        for idx, dim in enumerate(temp):
                            perm.append(dim)
                else:
                    for idx, dim in enumerate(temp):
                        perm.append(dim)

                tf_layers_dict[layer_id] = tf.transpose(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    perm=perm
                )

            ### Squeeze
            elif layer.attrib['type'] == 'Squeeze':
                axis = None
                if len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) == 1:
                    axis = int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])
                    if axis == 1:
                        axis = -1
                    elif axis >= 2:
                        axis -= 1
                else:
                    for idx, part_axis in enumerate(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]):
                        if part_axis == 1:
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)][idx] = -1
                        elif part_axis >= 2:
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)][idx] -= 1
                    axis = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                try:
                    tf_layers_dict[layer_id] = tf.squeeze(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        axis=axis
                    )
                except:
                    tf_layers_dict[layer_id] = tf.squeeze(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        axis=-1
                    )

            ### Gather
            elif layer.attrib['type'] == 'Gather':
                axis = int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)])
                temp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                input_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape[0]
                batch_dims = 0
                if not data is None and 'batch_dims' in data.attrib:
                    batch_dims = int(data.attrib['batch_dims'])
                    if batch_dims == 0:
                        batch_dims = None
                indices = []
                if type(temp) == np.ndarray:
                    for idx, dim in enumerate(temp):
                        indices.append(dim)
                else:
                    # TODO
                    shape = tf.shape(temp)
                    for idx, dim in enumerate(shape):
                        if idx == 0:
                            indices.append(0)
                        elif idx == input_shape - 1:
                            indices.append(1)
                        else:
                            indices.append(dim + 1)

                if isinstance(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], tf.Tensor):
                    tf_layers_dict[layer_id] = tf.gather(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        indices,
                        axis=axis,
                        batch_dims=batch_dims
                    )
                else:
                    if indices == [0] and axis == 0:
                        try:
                            tf_layers_dict[layer_id] = tf.squeeze(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                axis=axis
                            )
                            if tf_layers_dict[layer_id].type_spec.shape == []:
                                tf_layers_dict[layer_id] = tf.expand_dims(tf_layers_dict[layer_id], axis=0)
                        except:
                            tf_layers_dict[layer_id] = tf.gather(
                                tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                indices,
                                axis=axis,
                                batch_dims=batch_dims
                            )
                    else:
                        tf_layers_dict[layer_id] = tf.gather(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            indices,
                            axis=axis,
                            batch_dims=batch_dims
                        )

                    if batch_dims is None and axis == 0 and tf_layers_dict[layer_id].shape[0] == 1:
                        tf_layers_dict[layer_id] = tf_layers_dict[layer_id][0]

            ### GatherND
            elif layer.attrib['type'] == 'GatherND':
                batch_dims = data.attrib['batch_dims']
                params = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                indices_tmp = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                if indices_tmp.dtype is tf.float32 or indices_tmp.dtype is tf.float64:
                    indices = tf.cast(indices, tf.int64)
                else:
                    indices = indices_tmp
                tf_layers_dict[layer_id] = tf.gather_nd(params, indices, batch_dims=batch_dims)

            ### ReduceMean, ReduceMax, ReduceMin, ReduceSum, ReduceProd, ReduceL2 - TODO
            elif layer.attrib['type'] == 'ReduceMean' or layer.attrib['type'] == 'ReduceMax' or layer.attrib['type'] == 'ReduceMin' or \
                layer.attrib['type'] == 'ReduceSum' or layer.attrib['type'] == 'ReduceProd' or layer.attrib['type'] == 'ReduceL2':
                keep_dims = True if (data.attrib['keep_dims'] == "True" or data.attrib['keep_dims'] == "true") else False
                axis = None
                if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) == np.ndarray and len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) == 1:
                    axis = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].astype(np.int32)
                    if axis == 1:
                        axis = -1
                    elif axis >= 2:
                        axis -= 1
                elif type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) != np.ndarray and len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].shape) == 1:
                    try:
                        if (tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)].numpy() == [1, 2]).all():
                            if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]).dtype != tf.int32:
                                axis = tf.cast(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)], tf.int32)
                            else:
                                axis = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                        else:
                            if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]).dtype != tf.int32:
                                axis = tf.cast(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)] - 1, tf.int32)
                            else:
                                axis = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)] - 1
                    except:
                        if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]).dtype != tf.int32:
                            axis = tf.cast(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)] - 1, tf.int32)
                        else:
                            axis = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)] - 1
                else:
                    for idx, part_axis in enumerate(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]):
                        if part_axis == 1:
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)][idx] = -1
                        elif part_axis >= 2:
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)][idx] -= 1
                    if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]).dtype != tf.int32:
                        axis = tf.cast(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)], tf.int32)
                    else:
                        axis = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]

                if layer.attrib['type'] == 'ReduceMean':
                    tf_layers_dict[layer_id] = tf.math.reduce_mean(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        axis=axis,
                        keepdims=keep_dims
                    )
                elif layer.attrib['type'] == 'ReduceMax':
                    tf_layers_dict[layer_id] = tf.math.reduce_max(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        axis=axis,
                        keepdims=keep_dims
                    )
                elif layer.attrib['type'] == 'ReduceMin':
                    tf_layers_dict[layer_id] = tf.math.reduce_min(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        axis=axis,
                        keepdims=keep_dims
                    )
                elif layer.attrib['type'] == 'ReduceSum':
                    tf_layers_dict[layer_id] = tf.math.reduce_sum(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        axis=axis,
                        keepdims=keep_dims
                    )
                elif layer.attrib['type'] == 'ReduceProd':
                    tf_layers_dict[layer_id] = tf.math.reduce_prod(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        axis=axis,
                        keepdims=keep_dims
                    )
                elif layer.attrib['type'] == 'ReduceL2':
                    reduceL2_mul = tf.math.multiply(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                    )
                    reduceL2_sum = tf.math.reduce_sum(
                        reduceL2_mul,
                        axis=axis,
                        keepdims=keep_dims
                    )
                    tf_layers_dict[layer_id] = tf.math.rsqrt(reduceL2_sum)

            ### MatMul
            elif layer.attrib['type'] == 'MatMul':
                if not data is None and 'a' in data.attrib:
                    transpose_a = True if int(data.attrib['a']) == 1 else False
                if not data is None and 'b' in data.attrib:
                    transpose_b = True if int(data.attrib['b']) == 1 else False
                if not data is None and 'transpose_a' in data.attrib:
                    try:
                        transpose_a = True if int(data.attrib['transpose_a']) == 1 else False
                    except:
                        transpose_a = True if (data.attrib['transpose_a'] == 'True' or data.attrib['transpose_a'] == 'true') else False
                if not data is None and 'transpose_b' in data.attrib:
                    try:
                        transpose_b = True if int(data.attrib['transpose_b']) == 1 else False
                    except:
                        transpose_b = True if (data.attrib['transpose_b'] == 'True'or data.attrib['transpose_b'] == 'true') else False

                tf_layers_dict[layer_id] = tf.linalg.matmul(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                    transpose_a,
                    transpose_b
                )

            ### Reshape - TODO
            elif layer.attrib['type'] == 'Reshape':
                op1 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                op2 = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                op_type1 = type(op1)
                op_type2 = type(op2)
                op_len1 = len(np.asarray(op1.shape))
                op_len2 = len(np.asarray(op2.shape))

                shape = []
                if op_type1 != np.ndarray and op_type2 != np.ndarray:
                    # op and op
                    if op_len2 > 1:
                        shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]

                    elif op_len2 == 1 and op2.shape[0] == 2:
                        # TODO
                        shape = op2

                    elif op_len2 == 1 and op2.shape[0] == 3:
                        # TODO
                        shape = op2

                    elif op_len2 == 1 and op2.shape[0] == 4:
                        # TODO
                        shape = op2

                    elif op_len2 == 1 and op2.shape[0] == 5:
                        # TODO
                        shape = op2

                    elif op_len2 == 1 and op2.shape[0] == 6:
                        # YoloV4
                        shape = op2

                elif op_type1 != np.ndarray and op_type2 == np.ndarray:
                    # op and const
                    if op_len2 == 4:
                        op2 = op2.transpose(0,2,3,1)
                        shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                    elif op_len2 > 4:
                        shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                    elif op_len2 == 1 and op2.shape[0] == 1:
                        shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                    elif op_len2 == 1 and op2.shape[0] == 2:
                        shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                    elif op_len2 == 1 and op2.shape[0] == 3:
                        shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                        if op_len1 > len(op2):
                            if shape[2] == -1:
                                shape[0], shape[1], shape[2] = shape[0], shape[2], shape[1]

                    elif op_len2 == 1 and op2.shape[0] == 4:
                        one_count_1 = 0
                        for idx in op1.shape:
                            if idx == 1:
                                one_count_1 += 1
                        one_count_2 = 0
                        for idx in op2:
                            if idx == 1:
                                one_count_2 += 1
                        if one_count_1 != one_count_2 and one_count_2 == 3 and op2[3] != 1:
                            shape_tmp = []
                            shape_tmp.append(op2[0])
                            shape_tmp.append(op2[1])
                            shape_tmp.append(op2[2])
                            shape_tmp.append(op2[3])
                            shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(shape_tmp)]
                        else:
                            shape_tmp = []
                            shape_tmp.append(op2[0])
                            shape_tmp.append(op2[2])
                            shape_tmp.append(op2[3])
                            shape_tmp.append(op2[1])
                            shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(shape_tmp)]
                    elif op_len2 == 1 and op2.shape[0] == 5:
                        shape_tmp = []
                        if op2[1] == op2[2]:
                            shape_tmp.append(op2[0])
                            shape_tmp.append(op2[1])
                            shape_tmp.append(op2[2])
                            shape_tmp.append(op2[3])
                            shape_tmp.append(op2[4])
                        else:
                            shape_tmp.append(op2[0])
                            shape_tmp.append(op2[3])
                            shape_tmp.append(op2[4])
                            shape_tmp.append(op2[1])
                            shape_tmp.append(op2[2])

                        shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(shape_tmp)]
                    elif op_len2 == 1 and op2.shape[0] == 6:
                        # YoloV4
                        shape_tmp = []
                        shape_tmp.append(op2[0])
                        shape_tmp.append(op2[2])
                        shape_tmp.append(op2[3])
                        shape_tmp.append(op2[4])
                        shape_tmp.append(op2[5])
                        shape_tmp.append(op2[1])
                        shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(shape_tmp)]
                    else:
                        for i in range(op2.shape[0]):
                            shape.append(op2[i])

                elif op_type1 == np.ndarray and op_type2 != np.ndarray:
                    # const and op
                    if op_len1 == 4:
                        op1 = op1.transpose(0,2,3,1)
                        shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                    elif op_len1 > 4:
                        shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]
                    else:
                        for i in range(op2.shape[0]):
                            shape.append(op2[i])
                else:
                    # const and const
                    if op_len1 == 4:
                        op1 = op1.transpose(0,2,3,1)
                    if op_len2 == 4:
                        op2 = op2.transpose(0,2,3,1)
                    shape = [op1.shape[idx] if val == 0 else val for idx, val in enumerate(op2)]

                # tf_layers_dict[layer_id] = tf.reshape(op1, shape)
                temp = tf.transpose(op1, [0,3,1,2])
                tf_layers_dict[layer_id] = tf.reshape(temp, shape)

            ### Range - TODO
            elif layer.attrib['type'] == 'Range':
                dtype = cast_type_ov_tf[data.attrib['output_type']]
                start = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)][0]
                limit = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                try:
                    if start == 2 and 'Squeeze' in limit.name and type(limit.type_spec) == tf.TensorSpec and limit.type_spec.dtype == tf.int64:
                        start = 1
                        limit = tf.constant(3)
                except:
                    if start == 2 and limit.numpy() == 4:
                        start = 1
                        limit = tf.constant(3)
                tf_layers_dict[layer_id] = tf.range(
                    start,
                    limit,
                    delta=int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]),
                    dtype=dtype
                )

            ### Exp
            elif layer.attrib['type'] == 'Exp':
                tf_layers_dict[layer_id] = tf.math.exp(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Abs
            elif layer.attrib['type'] == 'Abs':
                tf_layers_dict[layer_id] = tf.math.abs(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### SoftMax
            elif layer.attrib['type'] == 'SoftMax':
                axis = int(data.attrib['axis'])
                if axis == 1 and len(np.asarray(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)) == 4:
                    axis = -1
                tf_layers_dict[layer_id] = tf.nn.softmax(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    axis=axis
                )

            ### Negative
            elif layer.attrib['type'] == 'Negative':
                tf_layers_dict[layer_id] = tf.math.negative(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Maximum
            elif layer.attrib['type'] == 'Maximum':
                # No broadcast
                tf_layers_dict[layer_id] = tf.math.maximum(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )

            ### Minimum
            elif layer.attrib['type'] == 'Minimum':
                # No broadcast
                tf_layers_dict[layer_id] = tf.math.minimum(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )

            ### Acos
            elif layer.attrib['type'] == 'Acos':
                tf_layers_dict[layer_id] = tf.math.acos(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Acosh
            elif layer.attrib['type'] == 'Acosh':
                tf_layers_dict[layer_id] = tf.math.acosh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Asin
            elif layer.attrib['type'] == 'Asin':
                tf_layers_dict[layer_id] = tf.math.asin(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Asinh
            elif layer.attrib['type'] == 'Asinh':
                tf_layers_dict[layer_id] = tf.math.asinh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Atan
            elif layer.attrib['type'] == 'Atan':
                tf_layers_dict[layer_id] = tf.math.atan(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Atanh
            elif layer.attrib['type'] == 'Atanh':
                tf_layers_dict[layer_id] = tf.math.atanh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Ceiling
            elif layer.attrib['type'] == 'Ceiling':
                tf_layers_dict[layer_id] = tf.math.ceil(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Cos
            elif layer.attrib['type'] == 'Cos':
                tf_layers_dict[layer_id] = tf.math.cos(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Cosh
            elif layer.attrib['type'] == 'Cosh':
                tf_layers_dict[layer_id] = tf.math.cosh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Sin
            elif layer.attrib['type'] == 'Sin':
                tf_layers_dict[layer_id] = tf.math.sin(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Sinh
            elif layer.attrib['type'] == 'Sinh':
                tf_layers_dict[layer_id] = tf.math.sinh(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Divide
            elif layer.attrib['type'] == 'Divide':
                x_np_type = None
                y_np_type = None

                x = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                if type(x) == np.ndarray and x.dtype == np.int:
                    x_np_type = tf.int32
                elif type(x) == np.ndarray:
                    x_np_type = tf.float32
                else:
                    x_np_type = x.dtype

                y = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                if type(y) == np.ndarray and y.dtype == np.int:
                    y_np_type = tf.int32
                elif type(y) == np.ndarray:
                    y_np_type = tf.float32
                else:
                    y_np_type = y.dtype

                if (x_np_type in int_type_tf) and (y_np_type in int_type_tf):
                    # floordiv
                    tf_layers_dict[layer_id] = tf.math.floordiv(x, y)
                else:
                    # divide
                    tf_layers_dict[layer_id] = tf.math.divide(x, y)

            ### Erf
            elif layer.attrib['type'] == 'Erf':
                tf_layers_dict[layer_id] = tf.math.erf(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Floor
            elif layer.attrib['type'] == 'Floor':
                tf_layers_dict[layer_id] = tf.math.floor(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### FloorMod
            elif layer.attrib['type'] == 'FloorMod':
                tf_layers_dict[layer_id] = tf.math.floormod(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )

            ### HSwish
            elif layer.attrib['type'] == 'HSwish':
                if replace_swish_and_hardswish:
                    # Swish
                    tf_layers_dict[layer_id] = tf.nn.swish(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                else:
                    # Hard-Swish
                    if not optimizing_hardswish_for_edgetpu:
                        tf_layers_dict[layer_id] = \
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] * \
                                tf.nn.relu6(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] + 3) * 0.16666667
                    else:
                        tf_layers_dict[layer_id] = \
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] * \
                                tf.nn.relu6(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] + 3) * 0.16666666

            ### Log
            elif layer.attrib['type'] == 'Log':
                tf_layers_dict[layer_id] = tf.math.log(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Power
            elif layer.attrib['type'] == 'Power':
                # No broadcast
                tf_layers_dict[layer_id] = tf.math.pow(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )

            ### Mish
            elif layer.attrib['type'] == 'Mish':
                tf_layers_dict[layer_id] = \
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)] * \
                        tf.math.tanh(tf.math.softplus(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]))

            ### Selu
            elif layer.attrib['type'] == 'Selu':
                tf_layers_dict[layer_id] = tf.nn.selu(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Subtract
            elif layer.attrib['type'] == 'Subtract':
                # No broadcast
                x = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                y = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                if type(x) == np.ndarray:
                    x = x.astype(np.float32)
                if type(y) == np.ndarray:
                    y = y.astype(np.float32)
                tf_layers_dict[layer_id] = tf.math.subtract(x, y)

            ### Unsqueeze - TODO
            elif layer.attrib['type'] == 'Unsqueeze':
                input_shape = np.asarray(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)
                indices = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]

                if len(input_shape) == 1 and indices == [0]:
                    tf_layers_dict[layer_id] = tf.identity(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                elif len(input_shape) > 1 and len(indices) > 1:
                    print('The multi-dimensional indices specification in Unsqueeze is not yet implemented.')
                    sys.exit(-1)
                else:
                    tf_layers_dict[layer_id] = tf.expand_dims(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        indices[0]
                    )

            ### Equal
            elif layer.attrib['type'] == 'Equal':
                tf_layers_dict[layer_id] = tf.math.equal(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )

            ### NotEqual
            elif layer.attrib['type'] == 'NotEqual':
                tf_layers_dict[layer_id] = tf.math.not_equal(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )

            ### Greater
            elif layer.attrib['type'] == 'Greater':
                tf_layers_dict[layer_id] = tf.math.greater(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )

            ### GreaterEqual
            elif layer.attrib['type'] == 'GreaterEqual':
                tf_layers_dict[layer_id] = tf.math.greater_equal(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )

            ### Less
            elif layer.attrib['type'] == 'Less':
                tf_layers_dict[layer_id] = tf.math.less(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )

            ### LessEqual
            elif layer.attrib['type'] == 'LessEqual':
                tf_layers_dict[layer_id] = tf.math.less_equal(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )

            ### Select
            elif layer.attrib['type'] == 'Select':
                tf_layers_dict[layer_id] = tf.raw_ops.SelectV2(
                    condition=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    t=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                    e=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]
                )

            ### LogicalAnd
            elif layer.attrib['type'] == 'LogicalAnd':
                tf_layers_dict[layer_id] = tf.math.logical_and(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )

            ### LogicalNot
            elif layer.attrib['type'] == 'LogicalNot':
                tf_layers_dict[layer_id] = tf.math.logical_not(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### LogicalOr
            elif layer.attrib['type'] == 'LogicalOr':
                tf_layers_dict[layer_id] = tf.math.logical_or(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )

            ### LogicalXor
            elif layer.attrib['type'] == 'LogicalXor':
                tf_layers_dict[layer_id] = tf.math.logical_xor(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )

            ### Broadcast - TODO
            elif layer.attrib['type'] == 'Broadcast':
                mode = data.attrib['mode']
                if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]) != np.ndarray:
                    if mode == 'numpy':
                        tf_layers_dict[layer_id] = tf.broadcast_to(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                        )
                    elif mode == 'bidirectional':
                        tf_layers_dict[layer_id] = tf.math.multiply(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            tf.ones(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])
                        )
                    else:
                        print(f'The {mode} mode of broadcast is not yet implemented.')
                        sys.exit(-1)
                else:
                    target_shape = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                    target_shape[0], target_shape[1], target_shape[2], target_shape[3] = \
                        target_shape[0], target_shape[2], target_shape[3], target_shape[1]
                    if mode == 'numpy':
                        tf_layers_dict[layer_id] = tf.broadcast_to(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            target_shape
                        )
                    elif mode == 'bidirectional':
                        tf_layers_dict[layer_id] = tf.math.multiply(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            tf.ones(target_shape)
                        )
                    else:
                        print(f'The {mode} mode of broadcast is not yet implemented.')
                        sys.exit(-1)

            ### Split
            elif layer.attrib['type'] == 'Split':
                num_splits = int(data.attrib['num_splits'])
                axis = int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])
                if axis == 1:
                    axis = 3
                elif axis >= 2:
                    axis -= 1

                def split_tensor(x, axis, num_split):
                    return tf.raw_ops.Split(axis=axis, value=x, num_split=num_split)

                outputs = Lambda(split_tensor,
                                arguments={'axis': axis,
                                            'num_split': num_splits}
                                )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

                for output, layer_id_port in zip(outputs, layer_id_port_dict[layer_id]['layer_id:port']):
                    tf_layers_dict[layer_id_port] = output

            ### VariadicSplit
            elif layer.attrib['type'] == 'VariadicSplit':
                axis = int(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)])
                input_shape_len = len(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape)

                if input_shape_len >= 4:
                    if axis == 1:
                        axis = -1
                    elif axis >= 2:
                        axis -= 1

                num_or_size_splits = None
                if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]) == np.ndarray:
                    num_or_size_splits = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]
                else:
                    num_or_size_splits = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]

                def split_tensor(x, axis, num_split):
                    return tf.raw_ops.Split(axis=axis, value=x, num_split=num_split)

                if len(num_or_size_splits) > 1 and np.average(num_or_size_splits) == num_or_size_splits[0]:
                    outputs = Lambda(split_tensor,
                                    arguments={
                                        'axis': axis,
                                        'num_split': len(num_or_size_splits)}
                                    )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])
                else:
                    if len(num_or_size_splits) > 1:
                        outputs = []

                        start_idx = 0
                        end_idx   = 0
                        split_start_list_all = []

                        for split_len in num_or_size_splits:
                            split_start_list_part = []
                            end_idx = start_idx + split_len - 1
                            for i in range(input_shape_len):
                                if axis == -1 and i == (input_shape_len - 1):
                                    split_start_list_part.append(start_idx)
                                elif axis == -1 and i != (input_shape_len - 1):
                                    split_start_list_part.append(0)
                                elif axis != -1 and i == axis:
                                    split_start_list_part.append(start_idx)
                                else:
                                    split_start_list_part.append(0)
                            split_start_list_all.append(split_start_list_part)
                            start_idx = end_idx + 1

                        split_size_list_all = []

                        for split_len in num_or_size_splits:
                            split_size_list_part = []
                            for i in range(input_shape_len):
                                if axis == -1 and i == (input_shape_len - 1):
                                    split_size_list_part.append(split_len)
                                elif axis == -1 and i != (input_shape_len - 1):
                                    split_size_list_part.append(-1)
                                elif axis != -1 and i == axis:
                                    split_size_list_part.append(split_len)
                                else:
                                    split_size_list_part.append(-1)
                            split_size_list_all.append(split_size_list_part)

                        for split_starts, split_sizes in zip(split_start_list_all, split_size_list_all):
                            outputs.append(
                                tf.slice(
                                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                                    split_starts, split_sizes
                                )
                            )

                    else:
                        outputs = tf.split(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            num_or_size_splits=num_or_size_splits,
                            axis=axis
                        )

                for output, layer_id_port in zip(outputs, layer_id_port_dict[layer_id]['layer_id:port']):
                    tf_layers_dict[layer_id_port] = output

            ### MVN - TODO axes
            elif layer.attrib['type'] == 'MVN':
                eps = float(data.attrib['eps'])
                across_channels = None
                if not data is None and 'across_channels' in data.attrib:
                    across_channels = data.attrib['across_channels']
                normalize_variance = None
                if not data is None and 'normalize_variance' in data.attrib:
                    normalize_variance = data.attrib['normalize_variance']
                eps_mode = None
                if not data is None and 'eps_mode' in data.attrib:
                    eps_mode = data.attrib['eps_mode']

                if across_channels == '0':
                    across_channels = False
                elif across_channels == '1':
                    across_channels = True
                elif across_channels == 'False' or across_channels == 'false':
                    across_channels = False
                elif across_channels == 'True' or across_channels == 'true':
                    across_channels = True

                if normalize_variance == '0':
                    normalize_variance = False
                elif normalize_variance == '1':
                    normalize_variance = True
                elif normalize_variance == 'False' or normalize_variance == 'false':
                    normalize_variance = False
                elif normalize_variance == 'True' or normalize_variance == 'true':
                    normalize_variance = True

                mean = None
                var = None
                axes = None
                data = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                try:
                    axes = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                except:
                    pass
                if across_channels:
                    mean = tf.math.reduce_mean(data, axis=[-1], keepdims=True)
                    var = tf.math.reduce_variance(data, axis=[-1], keepdims=True)
                else:
                    mean = tf.math.reduce_mean(data, axis=axes, keepdims=True)
                    var = tf.math.reduce_variance(data, axis=axes, keepdims=True)

                if normalize_variance:
                    if eps_mode == 'inside_sqrt':
                        mvn = (data - mean) / tf.math.sqrt(var + eps)
                    elif eps_mode == 'outside_sqrt':
                        mvn = (data - mean) / (tf.math.sqrt(var) + eps)
                    else:
                        mvn = (data - mean) / tf.math.sqrt(var + eps)
                else:
                    mvn = (data - mean)

                tf_layers_dict[layer_id] = mvn

            ### NonMaxSuppression - TODO
            elif layer.attrib['type'] == 'NonMaxSuppression':
                boxes = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)][0]
                batch_size = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].shape[0]
                if batch_size > 1:
                    print('When using NonMaxSuppression, fix the batch size to 1.')
                    sys.exit(-1)
                total_boxes_count = boxes.shape[0]
                scores = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)][0]
                class_count = scores.shape[0]

                max_output_boxes_per_class = 0
                try:
                    max_output_boxes_per_class = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)][0] # 25
                    if max_output_boxes_per_class == '-inf' or \
                        max_output_boxes_per_class == 'inf' or \
                            max_output_boxes_per_class == '-Infinity' or \
                                max_output_boxes_per_class == 'Infinity':
                        max_output_boxes_per_class = 0
                    elif max_output_boxes_per_class == 9223372036854775807:
                        max_output_boxes_per_class = total_boxes_count
                except:
                    pass

                iou_threshold = np.asarray(0.0, dtype=np.float32)
                try:
                    iou_threshold = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 3)][0] # 0.5
                    if iou_threshold == '-inf' or \
                        iou_threshold == 'inf' or \
                            iou_threshold == '-Infinity' or \
                                iou_threshold == 'Infinity':
                        iou_threshold = np.asarray(0.0, dtype=np.float32)
                    if type(iou_threshold) is np.float64:
                        iou_threshold = np.asarray(iou_threshold, dtype=np.float32)
                except:
                    pass

                score_threshold = np.asarray(0.0, dtype=np.float32)
                try:
                    score_threshold = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 4)][0] # -Infinity
                    if score_threshold == '-inf' or \
                        score_threshold == 'inf' or \
                            score_threshold == '-Infinity' or \
                                score_threshold == 'Infinity' or \
                                    score_threshold == float('-inf'):
                        score_threshold = np.asarray(0.0, dtype=np.float32)
                    if type(score_threshold) is np.float64:
                        score_threshold = np.asarray(score_threshold, dtype=np.float32)
                except:
                    pass

                soft_nms_sigma = np.asarray(0.0, dtype=np.float32)
                try:
                    soft_nms_sigma = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 5) ][0] # 0.1
                    if soft_nms_sigma == '-inf' or \
                        soft_nms_sigma == 'inf' or \
                            soft_nms_sigma == '-Infinity' or \
                                soft_nms_sigma == 'Infinity':
                        soft_nms_sigma = np.asarray(0.0, dtype=np.float32)
                except:
                    pass
                if type(soft_nms_sigma) is np.float64:
                    soft_nms_sigma = np.asarray(soft_nms_sigma, dtype=np.float32)
                box_encoding = data.attrib['box_encoding'] # corner or center
                output_type = data.attrib['output_type'] # i64 or i32
                sort_result_descending_str = data.attrib['sort_result_descending'] # True/true or False/false
                sort_result_descending = False
                if sort_result_descending_str == 'True' or sort_result_descending_str == 'true':
                    sort_result_descending = True

                if box_encoding == 'corner':

                    scores_tmp = tf.transpose(scores, perm=[1, 0])
                    score_top_values, score_top_idxes = tf.math.top_k(input=scores_tmp, k=1, sorted=False)
                    score_top_values_flat = tf.reshape(score_top_values, [-1])
                    score_top_idxes_flat = tf.reshape(score_top_idxes, [-1])
                    output_size = tf.math.minimum(total_boxes_count, max_output_boxes_per_class)

                    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
                        boxes=boxes,
                        scores=score_top_values_flat,
                        max_output_size=output_size,
                        iou_threshold=iou_threshold,
                        score_threshold=score_threshold,
                        soft_nms_sigma=soft_nms_sigma
                    )
                    selected_boxes = tf.gather(boxes, selected_indices)
                    selected_class_idx = tf.gather(score_top_idxes_flat, selected_indices)

                elif box_encoding == 'center':
                    print(f'The NonMaxSuppression box_encoding=center mode is not yet implemented.')
                    sys.exit(-1)

                tf_layers_dict['99990'] = tf.identity(
                    tf.reshape(selected_boxes, [batch_size, output_size, 4]),
                    name='selected_boxes'
                )
                tf_layers_dict['99991'] = tf.identity(
                    tf.reshape(selected_class_idx, [batch_size, output_size]),
                    name='selected_class_idx'
                )
                tf_layers_dict['99992'] = tf.identity(
                    tf.reshape(selected_scores, [batch_size, output_size]),
                    name='selected_scores'
                )
                tf_outputs.append(tf_layers_dict['99990'])
                tf_outputs.append(tf_layers_dict['99991'])
                tf_outputs.append(tf_layers_dict['99992'])

                tf_layers_dict[layer_id] = tf.zeros(
                    [output_size * class_count, 3],
                    name='dummy_non_max_suppression'
                )
                process_interruption_by_non_max_suppression = True

            ### NonZero
            elif layer.attrib['type'] == 'NonZero':
                output_type = tf.int64
                if not data is None and 'output_type' in data.attrib:
                    output_type = cast_type_ov_tf[data.attrib['output_type']]

                try:
                    # type_spec.dtype
                    if tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].type_spec.dtype != tf.bool:
                        input_type = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].type_spec.dtype
                        mask = tf.math.not_equal(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            tf.constant([0], dtype=input_type)
                        )
                        tf_layers_dict[layer_id] = tf.expand_dims(
                            tf.boolean_mask(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], mask),
                            axis=0
                        )
                    else:
                        temp_op = tf.cast(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], output_type)
                        mask = tf.math.not_equal(
                            temp_op,
                            tf.constant([0], dtype=output_type)
                        )
                        tf_layers_dict[layer_id] = tf.expand_dims(
                            tf.boolean_mask(temp_op, mask),
                            axis=0
                        )
                except:
                    # dtype
                    if tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].dtype != tf.bool:
                        input_type = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)].dtype
                        mask = tf.math.not_equal(
                            tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                            tf.constant([0], dtype=input_type)
                        )
                        tf_layers_dict[layer_id] = tf.expand_dims(
                            tf.boolean_mask(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], mask),
                            axis=0
                        )
                    else:
                        temp_op = tf.cast(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], output_type)
                        mask = tf.math.not_equal(
                            temp_op,
                            tf.constant([0], dtype=output_type)
                        )
                        tf_layers_dict[layer_id] = tf.expand_dims(
                            tf.boolean_mask(temp_op, mask),
                            axis=0
                        )

            ### SpaceToDepth
            elif layer.attrib['type'] == 'SpaceToDepth':
                block_size = int(data.attrib['block_size'])
                mode = data.attrib['mode']
                tf_layers_dict[layer_id] = tf.nn.space_to_depth(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    block_size=block_size
                )

            ### DepthToSpace
            elif layer.attrib['type'] == 'DepthToSpace':
                block_size = int(data.attrib['block_size'])
                mode = data.attrib['mode']

                def depth_to_space(x, block_size):
                    return tf.raw_ops.DepthToSpace(input=x, block_size=block_size)

                tf_layers_dict[layer_id] = Lambda(
                    depth_to_space,
                    arguments={'block_size': block_size}
                )(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)])

            ### Sqrt
            elif layer.attrib['type'] == 'Sqrt':
                tf_layers_dict[layer_id] = tf.math.sqrt(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )

            ### SquaredDifference
            elif layer.attrib['type'] == 'SquaredDifference':
                tf_layers_dict[layer_id] = tf.math.squared_difference(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )

            ### FakeQuantize
            elif layer.attrib['type'] == 'FakeQuantize':
                x = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)]
                x = tf.cast(x, dtype=tf.float32)
                if isinstance(x, tf.Tensor) and len(x.shape) == 4:
                    x = x.numpy()
                elif type(x) == np.ndarray and len(x.shape) == 4:
                    x = x

                input_low = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                input_low = tf.cast(input_low, dtype=tf.float32)
                if isinstance(input_low, tf.Tensor) and len(input_low.shape) == 4:
                    input_low = input_low.numpy().transpose(0,2,3,1)
                elif type(input_low) == np.ndarray and len(input_low.shape) == 4:
                    input_low = input_low.transpose(0,2,3,1)

                input_high = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 2)]
                input_high = tf.cast(input_high, dtype=tf.float32)
                if isinstance(input_high, tf.Tensor) and len(input_high.shape) == 4:
                    input_high = input_high.numpy().transpose(0,2,3,1)
                elif type(input_high) == np.ndarray and len(input_high.shape) == 4:
                    input_high = input_high.transpose(0,2,3,1)

                output_low = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 3)]
                output_low = tf.cast(output_low, dtype=tf.float32)
                if isinstance(output_low, tf.Tensor) and len(output_low.shape) == 4:
                    output_low = output_low.numpy().transpose(0,2,3,1)
                elif type(output_low) == np.ndarray and len(output_low.shape) == 4:
                    output_low = output_low.transpose(0,2,3,1)

                output_high = tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 4)]
                output_high = tf.cast(output_high, dtype=tf.float32)
                if isinstance(output_high, tf.Tensor) and len(output_high.shape) == 4:
                    output_high = output_high.numpy().transpose(0,2,3,1)
                elif type(output_high) == np.ndarray and len(output_high.shape) == 4:
                    output_high = output_high.transpose(0,2,3,1)

                levels = int(data.attrib['levels'])

                ### https://stackoverflow.com/questions/64111110/how-to-do-round-half-up-in-tensorflow
                ### https://www.xspdf.com/resolution/50434452.html
                tf_layers_dict[layer_id] = tf.where(tf.math.less_equal(x, tf.math.minimum(input_low, input_high)), output_low,
                                                    tf.where(tf.math.greater(x, tf.math.maximum(input_low, input_high)), output_high,
                                                    tf.floor(((x - input_low) / (input_high - input_low) * (levels-1)) + 0.5) / (levels-1) * (output_high - output_low) + output_low))

            ### Tile
            elif layer.attrib['type'] == 'Tile':
                tf_layers_dict[layer_id] = tf.tile(
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                    tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)]
                )

            ### Gelu
            elif layer.attrib['type'] == 'Gelu':
                approximation_mode = None
                if not data is None and 'approximation_mode' in data.attrib:
                    approximation_mode = data.attrib['approximation_mode']
                if approximation_mode == 'ERF' or approximation_mode is None:
                    tf_layers_dict[layer_id] = tf.nn.gelu(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        approximate=False
                    )
                elif approximation_mode == 'TANH':
                    tf_layers_dict[layer_id] = tf.nn.gelu(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        approximate=True
                    )

            ### NormalizeL2
            elif layer.attrib['type'] == 'NormalizeL2':
                eps = 1e-12 #None
                if not data is None and 'eps' in data.attrib:
                    eps = np.array(data.attrib['eps'], dtype=np.float32)
                eps_mode = None
                if not data is None and 'eps_mode' in data.attrib:
                    eps_mode = data.attrib['eps_mode']

                if eps_mode == 'add':
                    tf_layers_dict[layer_id] = tf.math.l2_normalize(
                        tf.math.add(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)], eps),
                        axis=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                        epsilon=0.0
                    )
                elif eps_mode == 'max':
                    tf_layers_dict[layer_id] = tf.math.l2_normalize(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        axis=tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 1)],
                        epsilon=eps
                    )

            ### Result
            elif layer.attrib['type'] == 'Result':
                if not process_interruption_by_non_max_suppression:
                    tf_layers_dict[layer_id] = tf.identity(
                        tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, 0)],
                        name=layer.attrib['name'].split('/')[0]
                    )
                    tf_outputs.append(tf_layers_dict[layer_id])
            else:
                print('The {} layer is not yet implemented.'.format(layer.attrib['type']))
                sys.exit(-1)

            if debug and idx > debug_layer_number:
                tf_outputs.append(tf_layers_dict[layer_id])
                break

        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            print(f'{Color.RED}ERROR:{Color.RESET} model_path  : {model_path}.xml')
            print(f'{Color.RED}ERROR:{Color.RESET} weights_path: {model_path}.bin')
            print(f'{Color.RED}ERROR:{Color.RESET} layer_id    :', layer_id)
            try:
                for edge_index in range(len(tf_edges[layer_id])):
                    if type(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, edge_index)]) != np.ndarray:
                        print(f'{Color.RED}ERROR:{Color.RESET} input_layer{edge_index} layer_id={tf_edges[layer_id][edge_index]}:', tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, edge_index)])
                    else:
                        print(f'{Color.RED}ERROR:{Color.RESET} input_layer{edge_index} layer_id={tf_edges[layer_id][edge_index]}: Const(ndarray).shape ', tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, edge_index)].shape)
                        pprint.pprint(tf_layers_dict[get_tf_edges_from(tf_edges, layer_id, edge_index)])
            except:
                pass
            print(f'{Color.RED}ERROR:{Color.RESET} The trace log is below.')
            import traceback
            traceback.print_exc()
            sys.exit(-1)

    # pprint.pprint(tf_outputs)
    model = Model(inputs=tf_inputs, outputs=tf_outputs)
    # model.summary()

    print(f'{Color.GREEN}TensorFlow/Keras model building process complete!{Color.RESET}')

    # saved_model output
    flag_for_output_switching_from_saved_model_to_pb_due_to_error = False
    if output_saved_model:
        try:
            print(f'{Color.REVERCE}saved_model output started{Color.RESET}', '=' * 58)
            tf.saved_model.save(model, model_output_path)
            # tf.keras.models.save_model(model, model_output_path, include_optimizer=False, save_format='tf', save_traces=False)
            # model.save(model_output_path, include_optimizer=False, save_format='tf', save_traces=False)
            print(f'{Color.GREEN}saved_model output complete!{Color.RESET}')
        except TypeError as e:
            print(f'{Color.GREEN}Switch to the output of an optimized protocol buffer file (.pb).{Color.RESET}')
            output_pb = True
            output_h5 = False
            flag_for_output_switching_from_saved_model_to_pb_due_to_error = True
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # .h5 output
    if output_h5:
        try:
            print(f'{Color.REVERCE}.h5 output started{Color.RESET}', '=' * 66)
            model.save(f'{model_output_path}/model_float32.h5', include_optimizer=False, save_format='h5')
            print(f'{Color.GREEN}.h5 output complete!{Color.RESET} - {model_output_path}/model_float32.h5')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # weight and json output
    if output_weight_and_json:
        try:
            print(f'{Color.REVERCE}weight and json output started{Color.RESET}', '=' * 54)
            open(f'{model_output_path}/model_float32.json', 'w').write(model.to_json())
            model.save_weights(f'{model_output_path}/model_float32_weights.h5')
            print(f'{Color.GREEN}weight and json output complete!{Color.RESET} - {model_output_path}/model_float32_weights.h5')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # .pb output
    if output_pb:
        try:
            print(f'{Color.REVERCE}.pb output started{Color.RESET}', '=' * 66)
            full_model = tf.function(lambda inputs: model(inputs))
            full_model = full_model.get_concrete_function(inputs=[tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])
            frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
            frozen_func.graph.as_graph_def()
            tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                                logdir=".",
                                name=f'{model_output_path}/model_float32.pb',
                                as_text=False)
            print(f'{Color.GREEN}.pb output complete!{Color.RESET} - {model_output_path}/model_float32.pb')

            if flag_for_output_switching_from_saved_model_to_pb_due_to_error:
                import shutil
                saved_model_tmp = 'saved_model_tmp'
                shutil.rmtree(saved_model_tmp, ignore_errors=True)
                os.makedirs(saved_model_tmp, exist_ok=True)
                inputs_tmp = []
                outputs_tmp = []
                for idx, _ in enumerate(model.inputs):
                    if idx == 0:
                        inputs_tmp.append(f'inputs:0')
                    else:
                        inputs_tmp.append(f'inputs_{idx}:0')
                for idx, _ in enumerate(model.outputs):
                    if idx == 0:
                        outputs_tmp.append(f'Identity:0')
                    else:
                        outputs_tmp.append(f'Identity_{idx}:0')

                def get_graph_def_from_file(graph_filepath):
                    tf.compat.v1.reset_default_graph()
                    tf.compat.v1.Graph().as_default()
                    with tf.compat.v1.gfile.GFile(graph_filepath, 'rb') as f:
                        graph_def = tf.compat.v1.GraphDef()
                        graph_def.ParseFromString(f.read())
                        return graph_def

                graph_def = get_graph_def_from_file(f'{model_output_path}/model_float32.pb')
                with tf.compat.v1.Session(graph=tf.Graph()) as sess:
                    tf.compat.v1.import_graph_def(graph_def, name='')
                    tf.compat.v1.saved_model.simple_save(
                        sess,
                        saved_model_tmp,
                        inputs= {t.rstrip(":0"):sess.graph.get_tensor_by_name(t) for t in inputs_tmp},
                        outputs={t.rstrip(":0"):sess.graph.get_tensor_by_name(t) for t in outputs_tmp}
                    )
                    from distutils.dir_util import copy_tree
                    copy_tree(saved_model_tmp, model_output_path)
                    shutil.rmtree(saved_model_tmp, ignore_errors=True)
                    print(f'{Color.GREEN}Optimized graph converted to SavedModel!{Color.RESET} - {model_output_path}')

        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # No Quantization - Input/Output=float32
    if output_no_quant_float32_tflite:
        try:
            print(f'{Color.REVERCE}tflite Float32 convertion started{Color.RESET}', '=' * 51)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            tflite_model = converter.convert()
            with open(f'{model_output_path}/model_float32.tflite', 'wb') as w:
                w.write(tflite_model)
            print(f'{Color.GREEN}tflite Float32 convertion complete!{Color.RESET} - {model_output_path}/model_float32.tflite')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # Weight Quantization - Input/Output=float32
    if output_weight_quant_tflite:
        try:
            print(f'{Color.REVERCE}Weight Quantization started{Color.RESET}', '=' * 57)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            tflite_model = converter.convert()
            with open(f'{model_output_path}/model_weight_quant.tflite', 'wb') as w:
                w.write(tflite_model)
            print(f'{Color.GREEN}Weight Quantization complete!{Color.RESET} - {model_output_path}/model_weight_quant.tflite')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # Float16 Quantization - Input/Output=float32
    if output_float16_quant_tflite:
        try:
            print(f'{Color.REVERCE}Float16 Quantization started{Color.RESET}', '=' * 56)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            tflite_quant_model = converter.convert()
            with open(f'{model_output_path}/model_float16_quant.tflite', 'wb') as w:
                w.write(tflite_quant_model)
            print(f'{Color.GREEN}Float16 Quantization complete!{Color.RESET} - {model_output_path}/model_float16_quant.tflite')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # Downloading datasets for calibration
    raw_test_data = None
    input_shapes = None
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
            print(f'{Color.REVERCE}numpy dataset load started{Color.RESET}', '=' * 58)
            try:
                if load_dest_file_path_for_the_calib_npy == npy_load_default_path and not os.path.exists(npy_load_default_path):
                    os.makedirs(os.path.dirname(npy_load_default_path), exist_ok=True)
                    import gdown
                    import subprocess
                    try:
                        result = subprocess.check_output(
                            [
                                'gdown',
                                '--id', '1z-K0KZCK3JBH9hXFuBTmIM4jaMPOubGN',
                                '-O', load_dest_file_path_for_the_calib_npy
                            ],
                            stderr=subprocess.PIPE
                        ).decode('utf-8')
                    except:
                        result = subprocess.check_output(
                            [
                                'sudo', 'gdown',
                                '--id', '1z-K0KZCK3JBH9hXFuBTmIM4jaMPOubGN',
                                '-O', load_dest_file_path_for_the_calib_npy
                            ],
                            stderr=subprocess.PIPE
                        ).decode('utf-8')
                raw_test_data = np.load(load_dest_file_path_for_the_calib_npy)
                print(f'{Color.GREEN}numpy dataset load complete!{Color.RESET}')
            except subprocess.CalledProcessError as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
                import traceback
                traceback.print_exc()
        else:
            pass
        input_shapes = [model_input.shape for model_input in model.inputs]

    def representative_dataset_gen():
        if calib_ds_type == 'tfds':
            for data in raw_test_data.take(10):
                image = data['image'].numpy()
                images = []
                for shape in input_shapes:
                    data = tf.image.resize(image, (shape[1], shape[2]))
                    tmp_image = eval(string_formulas_for_normalization) # Default: (data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]
                    tmp_image = tmp_image[np.newaxis,:,:,:]
                    images.append(tmp_image)
                yield images
        elif calib_ds_type == 'numpy':
            for idx in range(raw_test_data.shape[0]):
                image = raw_test_data[idx]
                images = []
                for shape in input_shapes:
                    data = tf.image.resize(image, (shape[1], shape[2]))
                    tmp_image = eval(string_formulas_for_normalization) # Default: (data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]
                    tmp_image = tmp_image[np.newaxis,:,:,:]
                    images.append(tmp_image)
                yield images

    # Integer Quantization
    if output_integer_quant_tflite:
        try:
            print(f'{Color.REVERCE}Integer Quantization started{Color.RESET}', '=' * 56)
            converter = tf.lite.TFLiteConverter.from_saved_model(model_output_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
            converter.representative_dataset = representative_dataset_gen
            tflite_model = converter.convert()
            with open(f'{model_output_path}/model_integer_quant.tflite', 'wb') as w:
                w.write(tflite_model)
            print(f'{Color.GREEN}Integer Quantization complete!{Color.RESET} - {model_output_path}/model_integer_quant.tflite')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # Full Integer Quantization
    if output_full_integer_quant_tflite:
        try:
            print(f'{Color.REVERCE}Full Integer Quantization started{Color.RESET}', '=' * 51)
            converter = tf.lite.TFLiteConverter.from_saved_model(model_output_path)
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
            with open(f'{model_output_path}/model_full_integer_quant.tflite', 'wb') as w:
                w.write(tflite_model)
            print(f'{Color.GREEN}Full Integer Quantization complete!{Color.RESET} - {model_output_path}/model_full_integer_quant.tflite')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # TensorFlow.js convert
    if output_tfjs:
        import subprocess
        try:
            print(f'{Color.REVERCE}TensorFlow.js Float32 convertion started{Color.RESET}', '=' * 44)
            result = subprocess.check_output(
                [
                    'tensorflowjs_converter',
                    '--input_format', 'tf_saved_model',
                    '--output_format', 'tfjs_graph_model',
                    '--signature_name', 'serving_default',
                    '--saved_model_tags', 'serve',
                    model_output_path, f'{model_output_path}/tfjs_model_float32'
                ],
                stderr=subprocess.PIPE
            ).decode('utf-8')
            print(result)
            print(f'{Color.GREEN}TensorFlow.js convertion complete!{Color.RESET} - {model_output_path}/tfjs_model_float32')
        except subprocess.CalledProcessError as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
            import traceback
            traceback.print_exc()
        try:
            print(f'{Color.REVERCE}TensorFlow.js Float16 convertion started{Color.RESET}', '=' * 44)
            result = subprocess.check_output(
                [
                    'tensorflowjs_converter',
                    '--quantize_float16',
                    '--input_format', 'tf_saved_model',
                    '--output_format', 'tfjs_graph_model',
                    '--signature_name', 'serving_default',
                    '--saved_model_tags', 'serve',
                    model_output_path, f'{model_output_path}/tfjs_model_float16'
                ],
                stderr=subprocess.PIPE
            ).decode('utf-8')
            print(result)
            print(f'{Color.GREEN}TensorFlow.js convertion complete!{Color.RESET} - {model_output_path}/tfjs_model_float16')
        except subprocess.CalledProcessError as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
            import traceback
            traceback.print_exc()

    # TF-TRT (TensorRT) convert
    if output_tftrt:
        try:
            def input_fn():
                input_shapes = []
                for tf_input in tf_inputs:
                    input_shapes.append(np.zeros(tf_input.shape).astype(np.float32))
                yield input_shapes

            print(f'{Color.REVERCE}TF-TRT (TensorRT) Float32 convertion started{Color.RESET}', '=' * 40)
            params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP32', maximum_cached_engines=10000)
            converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=model_output_path, conversion_params=params)
            converter.convert()
            converter.build(input_fn=input_fn)
            converter.save(f'{model_output_path}/tensorrt_saved_model_float32')
            print(f'{Color.GREEN}TF-TRT (TensorRT) convertion complete!{Color.RESET} - {model_output_path}/tensorrt_saved_model_float32')
            print(f'{Color.REVERCE}TF-TRT (TensorRT) Float16 convertion started{Color.RESET}', '=' * 40)
            params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP16', maximum_cached_engines=10000)
            converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=model_output_path, conversion_params=params)
            converter.convert()
            converter.build(input_fn=input_fn)
            converter.save(f'{model_output_path}/tensorrt_saved_model_float16')
            print(f'{Color.GREEN}TF-TRT (TensorRT) convertion complete!{Color.RESET} - {model_output_path}/tensorrt_saved_model_float16')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()
            print(f'{Color.RED}The binary versions of TensorFlow and TensorRT may not be compatible. Please check the version compatibility of each package.{Color.RESET}')

    # CoreML convert
    if output_coreml:
        try:
            print(f'{Color.REVERCE}CoreML convertion started{Color.RESET}', '=' * 59)
            mlmodel = ct.convert(model_output_path, source='tensorflow')
            mlmodel.save(f'{model_output_path}/model_coreml_float32.mlmodel')
            print(f'{Color.GREEN}CoreML convertion complete!{Color.RESET} - {model_output_path}/model_coreml_float32.mlmodel')
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

    # EdgeTPU convert
    if output_edgetpu:
        try:
            print(f'{Color.REVERCE}EdgeTPU convertion started{Color.RESET}', '=' * 58)
            result = subprocess.check_output(
                [
                    'edgetpu_compiler',
                    '-o', model_output_path,
                    '-sa',
                    f'{model_output_path}/model_full_integer_quant.tflite'
                ],
                stderr=subprocess.PIPE
            ).decode('utf-8')
            print(result)
            print(f'{Color.GREEN}EdgeTPU convert complete!{Color.RESET} - {model_output_path}/model_full_integer_quant_edgetpu.tflite')
        except subprocess.CalledProcessError as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
            import traceback
            traceback.print_exc()
            print("-" * 80)
            print('Please install edgetpu_compiler according to the following website.')
            print('https://coral.ai/docs/edgetpu/compiler/#system-requirements')

    # ONNX convert
    if output_onnx:
        import subprocess
        try:
            print(f'{Color.REVERCE}ONNX convertion started{Color.RESET}', '=' * 61)
            result = subprocess.check_output(
                [
                    'python3',
                    '-m', 'tf2onnx.convert',
                    '--saved-model', model_output_path,
                    '--opset', str(onnx_opset),
                    '--output', f'{model_output_path}/model_float32.onnx'
                ],
                stderr=subprocess.PIPE
            ).decode('utf-8')
            print(result)
            print(f'{Color.GREEN}ONNX convertion complete!{Color.RESET} - {model_output_path}/model_float32.onnx')
        except subprocess.CalledProcessError as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
            import traceback
            traceback.print_exc()

    # Myriad Inference Engine blob
    if output_myriad:
        try:
            print(f'{Color.REVERCE}Myriad Inference Engine blob convertion started{Color.RESET}', '=' * 44)
            os.makedirs(f'{model_output_path}/openvino/myriad', exist_ok=True)
            INTEL_OPENVINO_DIR = os.environ['INTEL_OPENVINO_DIR']
            result = subprocess.check_output(
                [
                    f'{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/myriad_compile',
                    '-m', f'{model_path}.xml',
                    '-VPU_NUMBER_OF_SHAVES', f'{vpu_number_of_shaves}',
                    '-VPU_NUMBER_OF_CMX_SLICES', f'{vpu_number_of_cmx_slices}',
                    '-o', f'{model_output_path}/openvino/myriad/saved_model.blob'
                ],
                stderr=subprocess.PIPE
            ).decode('utf-8')
            print(result)
            print(f'{Color.GREEN}Myriad Inference Engine blob convertion complete!{Color.RESET} - {model_output_path}/openvino/myriad')
        except subprocess.CalledProcessError as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e.stderr.decode('utf-8'))
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='input IR model path (.xml)')
    parser.add_argument('--model_output_path', type=str, default='saved_model', help='The output folder path of the converted model file')
    parser.add_argument('--output_saved_model', action='store_true', help='saved_model output switch')
    parser.add_argument('--output_h5', action='store_true', help='.h5 output switch')
    parser.add_argument('--output_weight_and_json', action='store_true', help='weight of h5 and json output switch')
    parser.add_argument('--output_pb', action='store_true', help='.pb output switch')
    parser.add_argument('--output_no_quant_float32_tflite', action='store_true', help='float32 tflite output switch')
    parser.add_argument('--output_weight_quant_tflite', action='store_true', help='weight quant tflite output switch')
    parser.add_argument('--output_float16_quant_tflite', action='store_true', help='float16 quant tflite output switch')
    parser.add_argument('--output_integer_quant_tflite', action='store_true', help='integer quant tflite output switch')
    parser.add_argument('--output_full_integer_quant_tflite', action='store_true', help='full integer quant tflite output switch')
    parser.add_argument('--output_integer_quant_type', type=str, default='int8', help='Input and output types when doing Integer Quantization (\'int8 (default)\' or \'uint8\')')
    parser.add_argument('--string_formulas_for_normalization', type=str, default='(data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]', help='String formulas for normalization. It is evaluated by Python\'s eval() function. Default: \'(data - [127.5,127.5,127.5]) / [127.5,127.5,127.5]\'')
    parser.add_argument('--calib_ds_type', type=str, default='numpy', help='Types of data sets for calibration. tfds or numpy. Only one of them can be specified. Default: numpy [20, 513, 513, 3] -> [Number of images, h, w, c]')
    parser.add_argument('--ds_name_for_tfds_for_calibration', type=str, default='coco/2017', help='Dataset name for TensorFlow Datasets for calibration. https://www.tensorflow.org/datasets/catalog/overview')
    parser.add_argument('--split_name_for_tfds_for_calibration', type=str, default='validation', help='Split name for TensorFlow Datasets for calibration. https://www.tensorflow.org/datasets/catalog/overview')
    tfds_dl_default_path = f'{str(Path.home())}/TFDS'
    parser.add_argument('--download_dest_folder_path_for_the_calib_tfds', type=str, default=tfds_dl_default_path, help='Download destination folder path for the calibration dataset. Default: $HOME/TFDS')
    parser.add_argument('--tfds_download_flg', action='store_true', help='True to automatically download datasets from TensorFlow Datasets. True or False')
    npy_load_default_path = 'sample_npy/calibration_data_img_sample.npy'
    parser.add_argument('--load_dest_file_path_for_the_calib_npy', type=str, default=npy_load_default_path, help='The path from which to load the .npy file containing the numpy binary version of the calibration data. Default: sample_npy/calibration_data_img_sample.npy')
    parser.add_argument('--output_tfjs', action='store_true', help='tfjs model output switch')
    parser.add_argument('--output_tftrt', action='store_true', help='tftrt model output switch')
    parser.add_argument('--output_coreml', action='store_true', help='coreml model output switch')
    parser.add_argument('--output_edgetpu', action='store_true', help='edgetpu model output switch')
    parser.add_argument('--output_onnx', action='store_true', help='onnx model output switch')
    parser.add_argument('--onnx_opset', type=int, default=13, help='onnx opset version number')
    parser.add_argument('--output_myriad', action='store_true', help='myriad inference engine blob output switch')
    parser.add_argument('--vpu_number_of_shaves', type=int, default=4, help='vpu number of shaves. Default: 4')
    parser.add_argument('--vpu_number_of_cmx_slices', type=int, default=4, help='vpu number of cmx slices. Default: 4')
    parser.add_argument('--replace_swish_and_hardswish', action='store_true', help='Replace swish and hard-swish with each other')
    parser.add_argument('--optimizing_hardswish_for_edgetpu', action='store_true', help='Optimizing hardswish for edgetpu')
    parser.add_argument('--replace_prelu_and_minmax', action='store_true', help='Replace prelu and minimum/maximum with each other')
    parser.add_argument('--yolact', action='store_true', help='Specify when converting the Yolact model')
    parser.add_argument('--restricted_resize_image_mode', action='store_true', help='Specify this if the upsampling contains OPs that are not scaled by integer multiples. Optimization for EdgeTPU will be disabled.')
    parser.add_argument('--weight_replacement_config', type=str, default='', help='Replaces the value of Const for each layer_id defined in json. Specify the path to the json file. "weight_replacement_config.json"')
    parser.add_argument('--debug', action='store_true', help='debug mode switch')
    parser.add_argument('--debug_layer_number', type=int, default=0, help='The last layer number to output when debugging. Used only when --debug=True')
    args = parser.parse_args()
    model, ext = os.path.splitext(args.model_path)
    model_output_path = args.model_output_path.rstrip('/')
    if ext != '.xml':
        print('The specified model is not \'.xml\' file.')
        sys.exit(-1)
    output_saved_model = args.output_saved_model
    output_h5 = args.output_h5
    output_weight_and_json = args.output_weight_and_json
    output_pb = args.output_pb
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
    load_dest_file_path_for_the_calib_npy = args.load_dest_file_path_for_the_calib_npy
    output_tfjs = args.output_tfjs
    output_tftrt = args.output_tftrt
    output_coreml = args.output_coreml
    output_edgetpu = args.output_edgetpu
    output_onnx = args.output_onnx
    onnx_opset = args.onnx_opset
    output_myriad = args.output_myriad
    vpu_number_of_shaves = args.vpu_number_of_shaves
    vpu_number_of_cmx_slices = args.vpu_number_of_cmx_slices
    replace_swish_and_hardswish = args.replace_swish_and_hardswish
    optimizing_hardswish_for_edgetpu = args.optimizing_hardswish_for_edgetpu
    replace_prelu_and_minmax = args.replace_prelu_and_minmax
    yolact = args.yolact
    restricted_resize_image_mode = args.restricted_resize_image_mode
    weight_replacement_config = args.weight_replacement_config
    debug = args.debug
    debug_layer_number = args.debug_layer_number
    if not output_saved_model and \
        not output_h5 and \
        not output_weight_and_json and \
        not output_pb and \
        not output_no_quant_float32_tflite and \
        not output_weight_quant_tflite and \
        not output_float16_quant_tflite and \
        not output_integer_quant_tflite and \
        not output_full_integer_quant_tflite and \
        not output_tfjs and \
        not output_tftrt and \
        not output_coreml and \
        not output_edgetpu and \
        not output_onnx and \
        not output_myriad:
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
    if output_onnx:
        if not 'tf2onnx' in package_list:
            print('\'tf2onnx\' is not installed. Please run the following command to install \'tf2onnx\'.')
            print('pip3 install --upgrade onnx')
            print('pip3 install --upgrade tf2onnx')
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
        pass
    else:
        print('Only \'tfds\' or \'numpy\' can be specified for calib_ds_type.')
        sys.exit(-1)

    if weight_replacement_config and not os.path.exists(weight_replacement_config):
        print('The json file does not exist in the path specified in weight_replacement_config.')
        sys.exit(-1)

    del package_list
    os.makedirs(model_output_path, exist_ok=True)
    convert(model, model_output_path, output_saved_model, output_h5, output_weight_and_json, output_pb,
            output_no_quant_float32_tflite, output_weight_quant_tflite, output_float16_quant_tflite,
            output_integer_quant_tflite, output_full_integer_quant_tflite, output_integer_quant_type,
            string_formulas_for_normalization,
            calib_ds_type, ds_name_for_tfds_for_calibration, split_name_for_tfds_for_calibration,
            download_dest_folder_path_for_the_calib_tfds, tfds_download_flg, npy_load_default_path, load_dest_file_path_for_the_calib_npy,
            output_tfjs, output_tftrt, output_coreml, output_edgetpu, output_onnx, onnx_opset, output_myriad,
            vpu_number_of_shaves, vpu_number_of_cmx_slices,
            replace_swish_and_hardswish, optimizing_hardswish_for_edgetpu, replace_prelu_and_minmax,
            yolact, restricted_resize_image_mode, weight_replacement_config, debug, debug_layer_number)
    print(f'{Color.REVERCE}All the conversion process is finished!{Color.RESET}', '=' * 45)

if __name__ == "__main__":
    main()