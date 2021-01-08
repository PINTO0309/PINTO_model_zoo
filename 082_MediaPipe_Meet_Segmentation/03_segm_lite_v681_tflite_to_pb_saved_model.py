### tf-nightly==2.5.0-dev20210104

### https://google.github.io/flatbuffers/flatbuffers_guide_tutorial.html

#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import json
import tensorflow.compat.v1 as tf
import tensorflow as tfv2
import shutil
from pathlib import Path
import pprint

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
schema = "schema.fbs"
binary = "./flatc"
model_path = "segm_lite_v681_opt.tflite"
output_pb_path = "segm_lite_v681_opt.pb"
output_savedmodel_path = "saved_model"
model_json_path = "segm_lite_v681_opt.json"
output_node_names = ['segment']

#################################################################
# Change to True when converting to EdgeTPU model.
optimizing_for_edgetpu_flg = False
#################################################################

def gen_model_json():
    if not os.path.exists(model_json_path):
        cmd = (binary + " -t --strict-json --defaults-json -o . {schema} -- {input}".format(input=model_path, schema=schema))
        print("output json command =", cmd)
        os.system(cmd)


def parse_json():
    j = json.load(open(model_json_path))
    op_types = [v['builtin_code'] for v in j['operator_codes']]
    print('op types:', op_types)
    ops = j['subgraphs'][0]['operators']
    print('num of ops:', len(ops))
    return ops, op_types

def optimizing_hardswish_for_edgetpu(input_op, name=None):
    ret_op = None
    if not optimizing_for_edgetpu_flg:
        ret_op = input_op * tf.nn.relu6(input_op + 3) * 0.16666667
    else:
        ret_op = input_op * tf.nn.relu6(input_op + 3) * 0.16666666
    return ret_op

def make_graph(ops, op_types, interpreter):

    height = 96
    width  = 160

    tensors = {}
    input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    print(input_details)
    for input_detail in input_details:
        tensors[input_detail['index']] = tf.placeholder(
            dtype=input_detail['dtype'],
            shape=input_detail['shape'],
            name=input_detail['name'])

    for op in ops:
        print('@@@@@@@@@@@@@@ op:', op)
        op_type = op_types[op['opcode_index']]
        if op_type == 'CONV_2D':
            input_tensor = tensors[op['inputs'][0]]
            weights = tensors[op['inputs'][1]].transpose(1,2,3,0)
            bias = tensors[op['inputs'][2]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            output_tensor = tf.nn.conv2d(
                input_tensor,
                weights,
                strides=[1, options['stride_h'], options['stride_w'], 1],
                padding=options['padding'],
                dilations=[
                    1, options['dilation_h_factor'],
                    options['dilation_w_factor'], 1
                ],
                name=output_detail['name'] + '/conv2d')
            output_tensor = tf.add(
                output_tensor, bias, name=output_detail['name'])

            if output_detail['name'].split('/')[-1]=='Relu6':
                output_tensor = tf.nn.relu6(output_tensor)

            tensors[output_detail['index']] = output_tensor
        elif op_type == 'DEPTHWISE_CONV_2D':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            input_tensor = tensors[op['inputs'][0]]
            weights = tensors[op['inputs'][1]].transpose(1,2,3,0)
            bias = tensors[op['inputs'][2]]
            options = op['builtin_options']
            output_tensor = tf.nn.depthwise_conv2d(
                input_tensor,
                weights,
                strides=[1, options['stride_h'], options['stride_w'], 1],
                padding=options['padding'],
                # dilations=[1, options['dilation_h_factor'], options['dilation_w_factor'], 1],
                name=output_detail['name'] + '/depthwise_conv2d')
            output_tensor = tf.add(output_tensor, bias, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'MAX_POOL_2D':
            input_tensor = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            output_tensor = tf.nn.max_pool(
                input_tensor,
                ksize=[
                    1, options['filter_height'], options['filter_width'], 1
                ],
                strides=[1, options['stride_h'], options['stride_w'], 1],
                padding=options['padding'],
                name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'PAD':
            input_tensor = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            paddings_detail = interpreter._get_tensor_details(op['inputs'][1])
            paddings_array = interpreter.get_tensor(paddings_detail['index'])
            paddings = tf.Variable(
                paddings_array, name=paddings_detail['name'])
            output_tensor = tf.pad(
                input_tensor, paddings, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'RELU':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            input_tensor = tensors[op['inputs'][0]]
            output_tensor = tf.nn.relu(input_tensor, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'PRELU':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            input_tensor = tensors[op['inputs'][0]]
            alpha_detail = interpreter._get_tensor_details(op['inputs'][1])
            alpha_array = interpreter.get_tensor(alpha_detail['index'])
            with tf.variable_scope(name_or_scope=output_detail['name']):
                alphas = tf.Variable(alpha_array, name=alpha_detail['name'])
                output_tensor = tf.maximum(alphas * input_tensor, input_tensor)
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'RELU6':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            input_tensor = tensors[op['inputs'][0]]
            output_tensor = tf.nn.relu6(input_tensor, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor 
        elif op_type == 'RESHAPE':
            input_tensor = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            output_tensor = tf.reshape(input_tensor, options['new_shape'], name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'ADD':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            input_tensor_0 = tensors[op['inputs'][0]]
            try:
                input_tensor_1 = tensors[op['inputs'][1]]
            except:
                param = interpreter._get_tensor_details(op['inputs'][1])
                input_tensor_1 = interpreter.get_tensor(param['index'])
            output_tensor = tf.add(input_tensor_0, input_tensor_1, name=output_detail['name'])

            if output_detail['name'].split('/')[-1]=='Relu6':
                output_tensor = tf.nn.relu6(output_tensor)

            tensors[output_detail['index']] = output_tensor
        elif op_type == 'CONCATENATION':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            input_tensor_0 = tensors[op['inputs'][0]]
            input_tensor_1 = tensors[op['inputs'][1]]
            try:
                input_tensor_2 = tensors[op['inputs'][2]]
                options = op['builtin_options']
                output_tensor = tf.concat([input_tensor_0, input_tensor_1, input_tensor_2],
                                        options['axis'],
                                        name=output_detail['name'])
            except:
                options = op['builtin_options']
                output_tensor = tf.concat([input_tensor_0, input_tensor_1],
                                        options['axis'],
                                        name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'LOGISTIC':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            input_tensor = tensors[op['inputs'][0]]
            output_tensor = tf.math.sigmoid(input_tensor, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'TRANSPOSE_CONV':
            input_tensor = tensors[op['inputs'][2]]
            weights_detail = interpreter._get_tensor_details(op['inputs'][1])
            output_shape_detail = interpreter._get_tensor_details(op['inputs'][0])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            weights_array = interpreter.get_tensor(weights_detail['index'])
            weights_array = np.transpose(weights_array, (1, 2, 0, 3))
            output_shape_array = interpreter.get_tensor(output_shape_detail['index'])
            weights = tf.Variable(weights_array, name=weights_detail['name'])
            shape = tf.Variable(output_shape_array, name=output_shape_detail['name'])
            options = op['builtin_options']
            output_tensor = tf.nn.conv2d_transpose(input_tensor,
                                                   weights,
                                                   shape,
                                                   [1, options['stride_h'], options['stride_w'], 1],
                                                   padding=options['padding'],
                                                   name=output_detail['name'] + '/conv2d_transpose')
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'MUL':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            input_tensor_0 = tensors[op['inputs'][0]]
            input_tensor_1 = None
            try:
                input_tensor_1 = tensors[op['inputs'][1]]
            except:
                param = interpreter._get_tensor_details(op['inputs'][1])
                input_tensor_1 = interpreter.get_tensor(param['index'])
            output_tensor = tf.multiply(input_tensor_0, input_tensor_1, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'HARD_SWISH':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            input_tensor = tensors[op['inputs'][0]]
            output_tensor = optimizing_hardswish_for_edgetpu(input_tensor, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'AVERAGE_POOL_2D':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            input_tensor = tensors[op['inputs'][0]]
            options = op['builtin_options']
            pool_size = [options['filter_height'], options['filter_width']]
            strides = [options['stride_h'], options['stride_w']]
            padding = options['padding']
            output_tensor = tf.keras.layers.AveragePooling2D(pool_size=pool_size,
                                                             strides=strides,
                                                             padding=padding,
                                                             name=output_detail['name'])(input_tensor)
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'FULLY_CONNECTED':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            input_tensor = tensors[op['inputs'][0]]
            weights = tensors[op['inputs'][1]].transpose(1,0)
            bias = tensors[op['inputs'][2]]
            output_shape_detail = interpreter._get_tensor_details(op['inputs'][0])
            output_shape_array = interpreter.get_tensor(output_shape_detail['index'])

            output_tensor = tf.keras.layers.Dense(units=output_shape_array.shape[3],
                                                  use_bias=True,
                                                  kernel_initializer=tf.keras.initializers.Constant(weights),
                                                  bias_initializer=tf.keras.initializers.Constant(bias))(input_tensor)
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'RESIZE_BILINEAR':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            input_tensor = tensors[op['inputs'][0]]
            size_detail = interpreter._get_tensor_details(op['inputs'][1])
            size = interpreter.get_tensor(size_detail['index'])
            size_height = size[0]
            size_width  = size[1]

            def upsampling2d_bilinear(x, size_height, size_width):
                if optimizing_for_edgetpu_flg:
                    return tf.image.resize_bilinear(x, (size_height, size_width))
                else:
                    return tfv2.image.resize(x, [size_height, size_width], method='bilinear')

            output_tensor = tf.keras.layers.Lambda(upsampling2d_bilinear, arguments={'size_height': size_height, 'size_width': size_width})(input_tensor)
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'DEQUANTIZE':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            weights_detail = interpreter._get_tensor_details(op['inputs'][0])
            weights = interpreter.get_tensor(weights_detail['index'])
            output_tensor = weights.astype(np.float32)
            tensors[output_detail['index']] = output_tensor
        else:
            raise ValueError(op_type)

    # Convolution2DTransposeBias
    input_tensor = tensors[241]
    weights = np.load('weights/segment_Kernel').transpose(1,2,0,3).astype(np.float32)
    bias = np.load('weights/segment_Bias').astype(np.float32)
    custom_trans = tf.nn.conv2d_transpose(input=input_tensor,
                                          filters=weights,
                                          output_shape=[1, height, width, 2],
                                          strides=[2, 2],
                                          padding='SAME',
                                          dilations=[1, 1])
    output_tensor = tf.math.add(custom_trans, bias, name='segment')
    tensors[999] = output_tensor


def main():

    tf.disable_eager_execution()

    gen_model_json()
    ops, op_types = parse_json()

    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)

    make_graph(ops, op_types, interpreter)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    graph = tf.get_default_graph()
    with tf.Session(config=config, graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=graph.as_graph_def(),
            output_node_names=output_node_names)

        with tf.io.gfile.GFile(output_pb_path, 'wb') as f:
            f.write(graph_def.SerializeToString())

        shutil.rmtree(output_savedmodel_path, ignore_errors=True)
        tf.saved_model.simple_save(
            sess,
            output_savedmodel_path,
            inputs={'input_1': graph.get_tensor_by_name('input_1:0')},
            outputs={'segment': graph.get_tensor_by_name('segment:0')}
        )

    converter = tfv2.lite.TFLiteConverter.from_saved_model(output_savedmodel_path)
    converter.target_spec.supported_ops = [tfv2.lite.OpsSet.TFLITE_BUILTINS, tfv2.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    with open(f'{output_savedmodel_path}/model_float32.tflite', 'wb') as w:
        w.write(tflite_model)

if __name__ == '__main__':
    main()

"""
$ saved_model_cli show --dir saved_model --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_1'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 144, 256, 3)
        name: input_1:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['segment'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 144, 256, 2)
        name: segment:0
  Method name is: tensorflow/serving/predict
"""
