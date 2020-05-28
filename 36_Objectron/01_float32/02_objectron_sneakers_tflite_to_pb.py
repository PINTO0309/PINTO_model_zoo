### tensorflow-gpu==1.15.2

### https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow
### https://www.tensorflow.org/lite/guide/ops_compatibility

#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import json
import tensorflow as tf
import shutil
from pathlib import Path
home = str(Path.home())

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
schema = "schema.fbs"
binary = home + "/flatc"
model_path = "object_detection_3d_sneakers.tflite"
output_pb_path = "object_detection_3d_sneakers.pb"
output_savedmodel_path = "saved_model_object_detection_3d_sneakers"
model_json_path = "object_detection_3d_sneakers.json"
num_tensors = 304
output_node_names = ['Identity', 'Identity_1', 'Identity_2']
# output_node_names = ['model/belief/Conv2D/conv2d', 'Identity_1', 'Identity_2']

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


def make_graph(ops, op_types, interpreter):
    tensors = {}
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    for input_detail in input_details:
        tensors[input_detail['index']] = tf.compat.v1.placeholder(
            dtype=input_detail['dtype'],
            shape=input_detail['shape'],
            name=input_detail['name'])

    for index, op in enumerate(ops):
        print('op: ', op)
        op_type = op_types[op['opcode_index']]
        if op_type == 'CONV_2D':
            input_tensor = tensors[op['inputs'][0]]
            weights_detail = interpreter._get_tensor_details(op['inputs'][1])
            bias_detail = interpreter._get_tensor_details(op['inputs'][2])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            # print('weights_detail: ', weights_detail)
            # print('bias_detail: ', bias_detail)
            # print('output_detail: ', output_detail)
            weights_array = interpreter.get_tensor(weights_detail['index'])
            weights_array = np.transpose(weights_array, (1, 2, 3, 0))
            bias_array = interpreter.get_tensor(bias_detail['index'])
            weights = tf.Variable(weights_array, name=weights_detail['name'])
            bias = tf.Variable(bias_array, name=bias_detail['name'])
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
            input_tensor = tensors[op['inputs'][0]]
            weights_detail = interpreter._get_tensor_details(op['inputs'][1])
            bias_detail = interpreter._get_tensor_details(op['inputs'][2])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            # print('weights_detail: ', weights_detail)
            # print('bias_detail: ', bias_detail)
            # print('output_detail: ', output_detail)
            weights_array = interpreter.get_tensor(weights_detail['index'])
            weights_array = np.transpose(weights_array, (1, 2, 3, 0))
            bias_array = interpreter.get_tensor(bias_detail['index'])
            weights = tf.Variable(weights_array, name=weights_detail['name'])
            bias = tf.Variable(bias_array, name=bias_detail['name'])
            options = op['builtin_options']
            output_tensor = tf.nn.depthwise_conv2d(
                input_tensor,
                weights,
                strides=[1, options['stride_h'], options['stride_w'], 1],
                padding=options['padding'],
                # dilations=[
                #     1, options['dilation_h_factor'],
                #     options['dilation_w_factor'], 1
                # ],
                name=output_detail['name'] + '/depthwise_conv2d')
            output_tensor = tf.add(
                output_tensor, bias, name=output_detail['name'])

            if output_detail['name'].split('/')[-1]=='Relu6':
                output_tensor = tf.nn.relu6(output_tensor)

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
            #print('input_tensor: ', input_tensor)
            #print('output_detail:', output_detail)
            #print('paddings_detail:', paddings_detail)
            paddings_array = interpreter.get_tensor(paddings_detail['index'])
            paddings = tf.Variable(
                paddings_array, name=paddings_detail['name'])
            output_tensor = tf.pad(
                input_tensor, paddings, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'RELU':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            input_tensor = tensors[op['inputs'][0]]
            output_tensor = tf.nn.relu(
                input_tensor, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'PRELU':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            input_tensor = tensors[op['inputs'][0]]
            alpha_detail = interpreter._get_tensor_details(op['inputs'][1])
            alpha_array = interpreter.get_tensor(alpha_detail['index'])
            with tf.compat.v1.variable_scope(name_or_scope=output_detail['name']):
                alphas = tf.Variable(alpha_array, name=alpha_detail['name'])
                output_tensor = tf.maximum(alphas * input_tensor, input_tensor)
            #print("PRELU.output_tensor=", output_tensor)
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'RESHAPE':
            input_tensor = tensors[op['inputs'][0]]
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            options = op['builtin_options']
            output_tensor = tf.reshape(
                input_tensor, options['new_shape'], name=output_detail['name'])
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
            #print('LOGISTIC op', op)
            #print('LOGISTIC output_detail:', output_detail)
            #print('LOGISTIC input_tensor:', input_tensor)
            output_tensor = tf.math.sigmoid(input_tensor, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'TRANSPOSE_CONV':
            input_tensor = tensors[op['inputs'][2]]
            weights_detail = interpreter._get_tensor_details(op['inputs'][1])
            output_shape_detail = interpreter._get_tensor_details(op['inputs'][0])
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            # print('********************* input_tensor: ', input_tensor)
            # print('weights_detail: ', weights_detail)
            # print('shape_detail: ', output_shape_detail)
            # print('output_detail: ', output_detail)
            weights_array = interpreter.get_tensor(weights_detail['index'])
            weights_array = np.transpose(weights_array, (1, 2, 0, 3))
            # print('weights_array.shape:', weights_array.shape)
            output_shape_array = interpreter.get_tensor(output_shape_detail['index'])
            # print('output_shape_array.shape:', output_shape_array.shape)
            weights = tf.Variable(weights_array, name=weights_detail['name'])
            shape = tf.Variable(output_shape_array, name=output_shape_detail['name'])
            options = op['builtin_options']
            output_tensor = tf.nn.conv2d_transpose(input_tensor,
                                                   weights,
                                                   shape,
                                                   [1, options['stride_h'], options['stride_w'], 1],
                                                   padding=options['padding'],
                                                   name=output_detail['name'] + '/conv2d_transpose')
            # print('********************* output_tensor:', output_tensor)
            tensors[output_detail['index']] = output_tensor
        elif op_type == 'MUL':
            output_detail = interpreter._get_tensor_details(op['outputs'][0])
            input_tensor_0 = tensors[op['inputs'][0]]
            param = interpreter._get_tensor_details(op['inputs'][1])
            input_tensor_1 = interpreter.get_tensor(param['index'])
            output_tensor = tf.multiply(input_tensor_0, input_tensor_1, name=output_detail['name'])
            tensors[output_detail['index']] = output_tensor
        else:
            raise ValueError(op_type)


def main():

    tf.compat.v1.disable_eager_execution()

    gen_model_json()
    ops, op_types = parse_json()

    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)
    for i in range(num_tensors):
        detail = interpreter._get_tensor_details(i)
        print(detail)

    make_graph(ops, op_types, interpreter)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    graph = tf.compat.v1.get_default_graph()
    # writer = tf.summary.FileWriter(os.path.splitext(output_pb_path)[0])
    # writer.add_graph(graph)
    # writer.flush()
    # writer.close()
    with tf.compat.v1.Session(config=config, graph=graph) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=graph.as_graph_def(),
            output_node_names=output_node_names)

        with tf.io.gfile.GFile(output_pb_path, 'wb') as f:
            f.write(graph_def.SerializeToString())

        shutil.rmtree('saved_model_object_detection_3d_sneakers', ignore_errors=True)
        tf.compat.v1.saved_model.simple_save(
            sess,
            output_savedmodel_path,
            inputs={'input': graph.get_tensor_by_name('input:0')},
            outputs={
                'Identity': graph.get_tensor_by_name('Identity:0'),
                # 'Identity': graph.get_tensor_by_name('model/belief/Conv2D/conv2d:0'),
                'Identity_1': graph.get_tensor_by_name('Identity_1:0'),
                'Identity_2': graph.get_tensor_by_name('Identity_2:0')
            })

if __name__ == '__main__':
    main()

"""
$ saved_model_cli show --dir saved_model_object_detection_3d_sneakers --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 640, 480, 3)
        name: input:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['Identity'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 40, 30, 1)
        name: model/belief/Conv2D/conv2d:0
    outputs['Identity_1'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 40, 30, 16)
        name: Identity_1:0
    outputs['Identity_2'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 160, 120, 4)
        name: Identity_2:0
  Method name is: tensorflow/serving/predict
"""
