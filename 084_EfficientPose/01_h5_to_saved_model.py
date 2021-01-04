import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from utils.helpers import Swish
from utils.helpers import swish1
from utils.helpers import eswish
from utils.helpers import keras_BilinearWeights
from tensorflow.keras import Model, Input
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
import pprint
import numpy as np
from pathlib import Path
import os


def convert(file_path, file_name, model_output_path, output_layer_names, mode, replace_table):

    model = tf.keras.models.load_model(f'{file_path}{file_name}.h5',
                                    custom_objects={'Swish': Swish,
                                                    'swish1': swish1,
                                                    'eswish': eswish,
                                                    'BilinearWeights': keras_BilinearWeights})

    new_model = None
    new_layers = None
    temp_layers = {}
    outputs = []

    for op in model.layers:
        # Input
        if isinstance(op, tf.keras.layers.InputLayer):
            # pprint.pprint(op.get_config())
            new_input = Input(shape=(op.input_shape[0][1], op.input_shape[0][2], op.input_shape[0][3]), batch_size=1, name=op.name)
            new_layers = new_input
            temp_layers[new_layers.name.split('/')[0]] = new_layers
            pprint.pprint(new_layers)
        elif isinstance(op, tf.keras.layers.Conv2DTranspose):
            pprint.pprint(op.get_config())
            new_layers = tf.keras.layers.Conv2DTranspose(filters=op.filters,
                                                        kernel_size=op.kernel_size,
                                                        strides=op.strides,
                                                        padding=op.padding,
                                                        output_padding=op.output_padding,
                                                        dilation_rate=op.dilation_rate,
                                                        activation=op.activation,
                                                        use_bias=op.use_bias,
                                                        kernel_initializer=op.kernel_initializer,
                                                        bias_initializer=op.bias_initializer,
                                                        name=op.name)(temp_layers[op.input.name.split('/')[0]])
            temp_layers[new_layers.name.split('/')[0]] = new_layers
            pprint.pprint(new_layers)
        elif isinstance(op, tf.keras.layers.DepthwiseConv2D):
            pprint.pprint(op.get_config())
            if op.use_bias:
                new_layers = tf.keras.layers.DepthwiseConv2D(kernel_size=op.kernel_size,
                                                            strides=op.strides,
                                                            padding=op.padding,
                                                            depth_multiplier=op.depth_multiplier,
                                                            dilation_rate=op.dilation_rate,
                                                            activation=op.activation,
                                                            use_bias=op.use_bias,
                                                            depthwise_initializer=tf.keras.initializers.Constant(op.weights[0].numpy()),
                                                            bias_initializer=tf.keras.initializers.Constant(op.bias),
                                                            trainable=False,
                                                            name=op.name)(temp_layers[op.input.name.split('/')[0]])
            else:
                new_layers = tf.keras.layers.DepthwiseConv2D(kernel_size=op.kernel_size,
                                                            strides=op.strides,
                                                            padding=op.padding,
                                                            depth_multiplier=op.depth_multiplier,
                                                            dilation_rate=op.dilation_rate,
                                                            activation=op.activation,
                                                            use_bias=op.use_bias,
                                                            depthwise_initializer=tf.keras.initializers.Constant(op.weights[0].numpy()),
                                                            trainable=False,
                                                            name=op.name)(temp_layers[op.input.name.split('/')[0]])

            temp_layers[new_layers.name.split('/')[0]] = new_layers
            pprint.pprint(new_layers)
        elif isinstance(op, tf.keras.layers.Conv2D):
            pprint.pprint(op.get_config())
            if op.use_bias:
                new_layers = tf.keras.layers.Conv2D(filters=op.filters,
                                                    kernel_size=op.kernel_size,
                                                    strides=op.strides,
                                                    padding=op.padding,
                                                    dilation_rate=op.dilation_rate,
                                                    groups=op.groups,
                                                    activation=op.activation,
                                                    use_bias=op.use_bias,
                                                    kernel_initializer=tf.keras.initializers.Constant(op.kernel),
                                                    bias_initializer=tf.keras.initializers.Constant(op.bias),
                                                    name=op.name)(temp_layers[op.input.name.split('/')[0]])
            else:
                new_layers = tf.keras.layers.Conv2D(filters=op.filters,
                                                    kernel_size=op.kernel_size,
                                                    strides=op.strides,
                                                    padding=op.padding,
                                                    dilation_rate=op.dilation_rate,
                                                    groups=op.groups,
                                                    activation=op.activation,
                                                    use_bias=op.use_bias,
                                                    kernel_initializer=tf.keras.initializers.Constant(op.kernel),
                                                    name=op.name)(temp_layers[op.input.name.split('/')[0]])



            temp_layers[new_layers.name.split('/')[0]] = new_layers
            pprint.pprint(new_layers)
        elif isinstance(op, tf.keras.layers.BatchNormalization):
            pprint.pprint(op.get_config())
            new_layers = tf.keras.layers.BatchNormalization(axis=op.axis,
                                                            momentum=op.momentum,
                                                            epsilon=op.epsilon,
                                                            center=op.center,
                                                            scale=op.scale,
                                                            beta_initializer=tf.keras.initializers.Constant(op.beta),
                                                            gamma_initializer=tf.keras.initializers.Constant(op.gamma),
                                                            moving_mean_initializer=tf.keras.initializers.Constant(op.moving_mean),
                                                            moving_variance_initializer=tf.keras.initializers.Constant(op.moving_variance),
                                                            trainable=False,
                                                            name=op.name)(temp_layers[op.input.name.split('/')[0]])
            temp_layers[new_layers.name.split('/')[0]] = new_layers
        elif isinstance(op, Swish):
            pprint.pprint(op.get_config())
            new_layers = Swish(activation=swish1,
                            name=op.name)(temp_layers[op.input.name.split('/')[0]])
            temp_layers[new_layers.name.split('/')[0]] = new_layers
            pprint.pprint(new_layers)
        elif isinstance(op, tf.keras.layers.AveragePooling2D):
            pprint.pprint(op.get_config())
            new_layers = tf.keras.layers.AveragePooling2D(pool_size=op.pool_size,
                                                        strides=op.strides,
                                                        padding=op.padding,
                                                        name=op.name)(temp_layers[op.input.name.split('/')[0]])
            temp_layers[new_layers.name.split('/')[0]] = new_layers
            pprint.pprint(new_layers)
        elif isinstance(op, tf.keras.layers.GlobalAveragePooling2D):
            pprint.pprint(op.get_config())
            new_layers = tf.keras.layers.GlobalAveragePooling2D(name=op.name)(temp_layers[op.input.name.split('/')[0]])
            temp_layers[new_layers.name.split('/')[0]] = new_layers
            pprint.pprint(new_layers)
        elif isinstance(op, tf.keras.layers.Reshape):
            pprint.pprint(op.get_config())
            new_layers = tf.keras.layers.Reshape(target_shape=op.target_shape, name=op.name)(temp_layers[op.input.name.split('/')[0]])
            temp_layers[new_layers.name.split('/')[0]] = new_layers
            pprint.pprint(new_layers)
        elif isinstance(op, tf.keras.layers.Multiply):
            pprint.pprint(op.get_config())
            pprint.pprint(op.input)
            input_layers = [temp_layers[input_op.name.split('/')[0]] for input_op in op.input]
            new_layers = tf.keras.layers.Multiply(name=op.name)(input_layers)
            temp_layers[new_layers.name.split('/')[0]] = new_layers
            pprint.pprint(new_layers)
        elif isinstance(op, tf.keras.layers.Lambda):
            pprint.pprint(op.get_config())
            new_layers = 1.25 * temp_layers[op.input.name.split('/')[0]] * tf.sigmoid(temp_layers[op.input.name.split('/')[0]])
            new_layers = tf.identity(new_layers, name=op.name.split('/')[0])
            temp_layers[op.name.split('/')[0]] = new_layers
            pprint.pprint(new_layers)
        elif isinstance(op, tf.keras.layers.Add):
            pprint.pprint(op.get_config())
            pprint.pprint(op.input)
            input_layers = []
            if op.input[0].name == 'Placeholder:0':
                input_layers.append(temp_layers[replace_table[op.input[1].name.split('/')[0]]])
                input_layers.append(temp_layers[op.input[1].name.split('/')[0]])
            elif op.input[1].name == 'Placeholder:0':
                input_layers.append(temp_layers[op.input[0].name.split('/')[0]])
                input_layers.append(temp_layers[replace_table[op.input[0].name.split('/')[0]]])
            else:
                input_layers.append(temp_layers[op.input[0].name.split('/')[0]])
                input_layers.append(temp_layers[op.input[1].name.split('/')[0]])
            new_layers = tf.keras.layers.Add(name=op.name)(input_layers)
            temp_layers[new_layers.name.split('/')[0]] = new_layers
            pprint.pprint(new_layers)
        elif isinstance(op, tf.keras.layers.Concatenate):
            pprint.pprint(op.get_config())
            pprint.pprint(op.input)
            input_layers = []
            input_len = len(op.input)
            if input_len == 2:
                if op.input[0].name == 'Placeholder:0':
                    input_layers.append(temp_layers[replace_table[op.input[1].name.split('/')[0]]])
                    input_layers.append(temp_layers[op.input[1].name.split('/')[0]])
                elif op.input[1].name == 'Placeholder:0':
                    input_layers.append(temp_layers[op.input[0].name.split('/')[0]])
                    input_layers.append(temp_layers[replace_table[op.input[0].name.split('/')[0]]])
                else:
                    input_layers.append(temp_layers[op.input[0].name.split('/')[0]])
                    input_layers.append(temp_layers[op.input[1].name.split('/')[0]])
            elif input_len == 3:
                if op.input[0].name == 'Placeholder:0':
                    input_layers.append(temp_layers[replace_table[op.input[1].name.split('/')[0]]])
                    input_layers.append(temp_layers[op.input[1].name.split('/')[0]])
                    input_layers.append(temp_layers[op.input[2].name.split('/')[0]])
                elif op.input[1].name == 'Placeholder:0':
                    input_layers.append(temp_layers[op.input[0].name.split('/')[0]])
                    input_layers.append(temp_layers[replace_table[op.input[0].name.split('/')[0]]])
                    input_layers.append(temp_layers[op.input[2].name.split('/')[0]])
                elif op.input[2].name == 'Placeholder:0':
                    try:
                        input_layers.append(temp_layers[op.input[0].name.split('/')[0]])
                        input_layers.append(temp_layers[op.input[1].name.split('/')[0]])
                        input_layers.append(temp_layers[replace_table[op.input[0].name.split('/')[0]]])
                    except:
                        input_layers = []
                        input_layers.append(temp_layers[op.input[0].name.split('/')[0]])
                        input_layers.append(temp_layers[op.input[1].name.split('/')[0]])
                        input_layers.append(temp_layers[replace_table[op.input[1].name.split('/')[0]]])
                else:
                    input_layers.append(temp_layers[op.input[0].name.split('/')[0]])
                    input_layers.append(temp_layers[op.input[1].name.split('/')[0]])
                    input_layers.append(temp_layers[op.input[2].name.split('/')[0]])
            else:
                import sys
                sys.exit(1)
            new_layers = tf.keras.layers.Concatenate(axis=op.axis, name=op.name)(input_layers)
            temp_layers[new_layers.name.split('/')[0]] = new_layers
            pprint.pprint(new_layers)
        elif isinstance(op, tf.keras.layers.Activation):
            pprint.pprint(op.get_config())
            new_layers = tf.keras.layers.Activation(activation=op.activation, name=op.name)(temp_layers[op.input.name.split('/')[0]])
            temp_layers[new_layers.name.split('/')[0]] = new_layers
            pprint.pprint(new_layers)
        else:
            print(op)
            break
        # output mem
        if new_layers.name.split('/')[0] in output_layer_names:
            outputs.append(new_layers)




    new_model = Model(inputs=new_input, outputs=outputs)
    new_model.summary()

    # # saved_model
    # tf.saved_model.save(new_model, model_output_path)

    if mode == 'gen_pb':
        # .pb
        full_model = tf.function(lambda inputs: new_model(inputs))
        full_model = full_model.get_concrete_function(inputs=[tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in new_model.inputs])
        frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                            logdir=".",
                            name=f'{model_output_path}/model_float32.pb',
                            as_text=False)
    
    else:
        ##########################################################################
        # .pb to saved_model convert
        #
        # $ sudo pip3 install openvino2tensorflow --upgrade
        # pb_to_saved_model \
        # --pb_file_path saved_model_opt_EfficientPoseI/model_float32.pb \
        # --inputs inputs:0 \
        # --outputs Identity:0,Identity_1:0,Identity_2:0,Identity_3:0
        #
        # pb_to_saved_model \
        # --pb_file_path saved_model_opt_EfficientPoseII/model_float32.pb \
        # --inputs inputs:0 \
        # --outputs Identity:0,Identity_1:0,Identity_2:0,Identity_3:0 
        #
        # pb_to_saved_model \
        # --pb_file_path saved_model_opt_EfficientPoseIII/model_float32.pb \
        # --inputs inputs:0 \
        # --outputs Identity:0,Identity_1:0,Identity_2:0,Identity_3:0 
        #
        # pb_to_saved_model \
        # --pb_file_path saved_model_opt_EfficientPoseIV/model_float32.pb \
        # --inputs inputs:0 \
        # --outputs Identity:0,Identity_1:0,Identity_2:0,Identity_3:0
        ##########################################################################

        # No Quantization - Input/Output=float32
        converter = tf.lite.TFLiteConverter.from_saved_model(model_output_path)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        with open(f'{model_output_path}/model_float32.tflite', 'wb') as w:
            w.write(tflite_model)

        # Weight Quantization - Input/Output=float32
        converter = tf.lite.TFLiteConverter.from_saved_model(model_output_path)
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        with open(f'{model_output_path}/model_weight_quant.tflite', 'wb') as w:
            w.write(tflite_model)

        # Float16 Quantization - Input/Output=float32
        converter = tf.lite.TFLiteConverter.from_saved_model(model_output_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_quant_model = converter.convert()
        with open(f'{model_output_path}/model_float16_quant.tflite', 'wb') as w:
            w.write(tflite_quant_model)

        # Downloading datasets for calibration
        raw_test_data = None
        input_shapes = None
        raw_test_data = tfds.load(name='coco/2017',
                                with_info=False,
                                split='validation',
                                data_dir=f'{str(Path.home())}/TFDS',
                                download=True)
        input_shapes = [model_input.shape for model_input in model.inputs]

        def representative_dataset_gen():
            for data in raw_test_data.take(10):
                image = data['image'].numpy()
                images = []
                for shape in input_shapes:
                    data = tf.image.resize(image, (shape[1], shape[2]))
                    tmp_image = eval('data / 255')
                    tmp_image = tmp_image[np.newaxis,:,:,:]
                    images.append(tmp_image)
                yield images

        # Integer Quantization
        converter = tf.lite.TFLiteConverter.from_saved_model(model_output_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.representative_dataset = representative_dataset_gen
        tflite_model = converter.convert()
        with open(f'{model_output_path}/model_integer_quant.tflite', 'wb') as w:
            w.write(tflite_model)

        # Full Integer Quantization
        converter = tf.lite.TFLiteConverter.from_saved_model(model_output_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
        # inf_type = tf.uint8
        inf_type = tf.int8
        converter.inference_input_type = inf_type
        converter.inference_output_type = inf_type
        converter.representative_dataset = representative_dataset_gen
        tflite_model = converter.convert()
        with open(f'{model_output_path}/model_full_integer_quant.tflite', 'wb') as w:
            w.write(tflite_model)

if __name__ == "__main__":

    mode = 'gen_pb' # 'gen_pb' or 'quantization'

    # EfficientPoseI
    file_path = 'models/keras/'
    file_name = 'EfficientPoseI'
    model_output_path = f'saved_model_opt_{file_name}'
    output_layer_names = ['pass1_skeleton_pafs','pass2_detection1_confs','pass3_detection2_confs','upscaled_confs']
    replace_table = {'block1a_project_bn_res1': 'lambda_1_res1',
                     'block2a_project_bn_res1': 'lambda_2_res1',
                     'block2b_add_res1': 'lambda_3_res1',
                     'block3a_project_bn_res1': 'lambda_4_res1',
                     'block3b_add_res1': 'lambda_5_res1',
                     'block3c_add_res1': 'lambda_17_res2'}
    os.makedirs(model_output_path, exist_ok=True)
    convert(file_path, file_name, model_output_path, output_layer_names, mode, replace_table)
    print('@@@@@@@@@@@@@@@@@@@@@@ EfficientPoseI complete @@@@@@@@@@@@@@@@@@@@@@')

    # EfficientPoseII
    file_path = 'models/keras/'
    file_name = 'EfficientPoseII'
    model_output_path = f'saved_model_opt_{file_name}'
    output_layer_names = ['pass1_skeleton_pafs','pass2_detection1_confs','pass3_detection2_confs','upscaled_confs']
    replace_table = {'block1a_project_bn_res1': 'lambda_1_res1',
                     'block2a_project_bn_res1': 'lambda_2_res1',
                     'block2b_add_res1': 'lambda_3_res1',
                     'block2c_add_res1': 'lambda_4_res1',
                     'block3a_project_bn_res1': 'lambda_5_res1',
                     'block3b_add_res1': 'lambda_6_res1',
                     'block3c_add_res1': 'lambda_7_res1',
                     'block3d_add_res1': 'lambda_26_res2'}
    os.makedirs(model_output_path, exist_ok=True)
    convert(file_path, file_name, model_output_path, output_layer_names, mode, replace_table)
    print('@@@@@@@@@@@@@@@@@@@@@@ EfficientPoseII complete @@@@@@@@@@@@@@@@@@@@@@')

    # EfficientPoseIII
    file_path = 'models/keras/'
    file_name = 'EfficientPoseIII'
    model_output_path = f'saved_model_opt_{file_name}'
    output_layer_names = ['pass1_skeleton_pafs','pass2_detection1_confs','pass3_detection2_confs','upscaled_confs']
    replace_table = {'block1b_add_res1': 'lambda_2_res1',
                     'block1a_project_bn_res1': 'lambda_1_res1',
                     'block2a_project_bn_res1': 'lambda_3_res1',
                     'block2b_add_res1': 'lambda_4_res1',
                     'block2c_add_res1': 'lambda_5_res1',
                     'block2d_add_res1': 'lambda_6_res1',
                     'block3a_project_bn_res1': 'lambda_7_res1',
                     'block1a_project_bn_res2': 'lambda_33_res2',
                     'block3b_add_res1': 'lambda_8_res1',
                     'block3c_add_res1': 'lambda_9_res1',
                     'block2a_project_bn_res2': 'lambda_34_res2',
                     'block3d_add_res1': 'lambda_10_res1',
                     'block3e_add_res1': 'lambda_35_res2'}
    os.makedirs(model_output_path, exist_ok=True)
    convert(file_path, file_name, model_output_path, output_layer_names, mode, replace_table)
    print('@@@@@@@@@@@@@@@@@@@@@@ EfficientPoseIII complete @@@@@@@@@@@@@@@@@@@@@@')

    # EfficientPoseIV
    file_path = 'models/keras/'
    file_name = 'EfficientPoseIV'
    model_output_path = f'saved_model_opt_{file_name}'
    output_layer_names = ['pass1_skeleton_pafs','pass2_detection1_confs','pass3_detection2_confs','upscaled_confs']
    replace_table = {'block1a_project_bn_res1': 'lambda_1_res1',
                     'block1b_add_res1': 'lambda_2_res1',
                     'block1c_add_res1': 'lambda_3_res1',
                     'block2a_project_bn_res1': 'lambda_4_res1',
                     'block2b_add_res1': 'lambda_5_res1',
                     'block2c_add_res1': 'lambda_6_res1',
                     'block2d_add_res1': 'lambda_7_res1',
                     'block2e_add_res1': 'lambda_8_res1',
                     'block2f_add_res1': 'lambda_9_res1',
                     'block3a_project_bn_res1': 'lambda_10_res1',
                     'block3b_add_res1': 'lambda_11_res1',
                     'block3c_add_res1': 'lambda_12_res1',
                     'block1a_project_bn_res2': 'lambda_49_res2',
                     'block3d_add_res1': 'lambda_13_res1',
                     'block3e_add_res1': 'lambda_14_res1',
                     'block2a_project_bn_res2': 'lambda_50_res2',
                     'block3f_add_res1': 'lambda_15_res1',
                     'block3g_add_res1': 'lambda_51_res2'}
    os.makedirs(model_output_path, exist_ok=True)
    convert(file_path, file_name, model_output_path, output_layer_names, mode, replace_table)
    print('@@@@@@@@@@@@@@@@@@@@@@ EfficientPoseIV complete @@@@@@@@@@@@@@@@@@@@@@')