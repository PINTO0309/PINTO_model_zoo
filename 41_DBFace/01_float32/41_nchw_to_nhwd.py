import tensorflow.compat.v1 as tf
from tensorflow.python import ops
import shutil

graph_filepath = 'dbface.pb'
tf.reset_default_graph()
with ops.Graph().as_default():
    with tf.gfile.GFile(graph_filepath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

graph = tf.get_default_graph()

with tf.Session(graph=graph) as session:

    # Examine the OP in NCHW format
    tf.import_graph_def(graph_def, name='')
    target_ops = []
    for op in graph.get_operations():
        if 'data_format' in op.node_def.attr and op.node_def.attr['data_format'].s == b'NCHW':
            target_ops.append(op)

    # for target_op in target_ops:
    #     print(f'op: {target_op.name} ({target_op.type})')

    # ---- NCHW to NHWC Convert
    # 1. Conv2D
    # 2. FusedBatchNormV3
    # 3. DepthwiseConv2dNative
    # 4. BiasAdd
    nodes = []
    before_node = None
    for target_op in target_ops:
        # Inputs
        if target_op.type in ['Conv2D','DepthwiseConv2dNative','BiasAdd']:
            inputs = [
                tf.transpose(
                    target_op.inputs[0],
                    [0, 2, 3, 1],
                    name=f'{target_op.name}_input_transpose'),
                target_op.inputs[1]
            ]
        elif target_op.type == 'FusedBatchNormV3':
            inputs = [
                tf.transpose(
                    target_op.inputs[0],
                    [0, 2, 3, 1],
                    name=f'{target_op.name}_input_transpose'),
                target_op.inputs[1],
                target_op.inputs[2],
                target_op.inputs[3],
                target_op.inputs[4]
            ]

        # Attributes
        attrs = {}
        for k, v in target_op.node_def.attr.items():
            if k == 'data_format':
                continue
            elif k == 'use_cudnn_on_gpu':
                attrs[k] = tf.AttrValue(b=False)
                continue
            if target_op.type in ['Conv2D','DepthwiseConv2dNative'] and k == 'strides':
                strides = v.list.i
                attrs[k] = tf.AttrValue(list=tf.AttrValue.ListValue(i=[
                    strides[0],
                    strides[2],
                    strides[3],
                    strides[1],
                ]))
            else:
                attrs[k] = v

        # New operations
        new_op = graph.create_op(op_type=target_op.type, inputs=inputs, name=f'{target_op.name}_nhwc', attrs=attrs)
        output = tf.transpose(new_op.outputs[0], [0, 3, 1, 2], name=f'{new_op.name}_output')
        # Update connections
        ops = [op for op in graph.get_operations() if target_op.outputs[0] in op.inputs]
        for op in ops:
            for i, input_tensor in enumerate(op.inputs):
                if input_tensor.name == target_op.outputs[0].name:
                    op._update_input(i, output)

    # Save model (1)
    with open('dbface_nhwc.pb', 'wb') as f:
        f.write(session.graph.as_graph_def().SerializeToString())
    print('NCHW to NHWC converted! (Freeze Graph - dbface_nhwc.pb)')

    # Save model (2)
    shutil.rmtree('saved_model_nhwc', ignore_errors=True)
    tf.enable_resource_variables()
    tf.saved_model.simple_save(
        session,
        'saved_model_nhwc',
        inputs={'x': session.graph.get_tensor_by_name('x:0')},
        outputs={t.rstrip(":0"):session.graph.get_tensor_by_name(t) for t in ['Identity:0','Identity_1:0','Identity_2:0']}
    )
    print('NCHW to NHWC converted! (saved_model - saved_model_nhwc)')


"""
op: model/475/Conv2D (Conv2D)
op: model/476/FusedBatchNormV3 (FusedBatchNormV3)
op: model/478/Conv2D (Conv2D)
op: model/479/FusedBatchNormV3 (FusedBatchNormV3)
op: model/481/depthwise (DepthwiseConv2dNative)
op: model/482/FusedBatchNormV3 (FusedBatchNormV3)
op: model/484/Conv2D (Conv2D)
op: model/485/FusedBatchNormV3 (FusedBatchNormV3)
op: model/487/Conv2D (Conv2D)
op: model/488/FusedBatchNormV3 (FusedBatchNormV3)
op: model/490/depthwise (DepthwiseConv2dNative)
op: model/491/FusedBatchNormV3 (FusedBatchNormV3)
op: model/493/Conv2D (Conv2D)
op: model/494/FusedBatchNormV3 (FusedBatchNormV3)
op: model/495/Conv2D (Conv2D)
op: model/496/FusedBatchNormV3 (FusedBatchNormV3)
op: model/498/depthwise (DepthwiseConv2dNative)
op: model/499/FusedBatchNormV3 (FusedBatchNormV3)
op: model/501/Conv2D (Conv2D)
op: model/502/FusedBatchNormV3 (FusedBatchNormV3)
op: model/504/Conv2D (Conv2D)
op: model/505/FusedBatchNormV3 (FusedBatchNormV3)
op: model/507/depthwise (DepthwiseConv2dNative)
op: model/508/FusedBatchNormV3 (FusedBatchNormV3)
op: model/510/Conv2D (Conv2D)
op: model/511/FusedBatchNormV3 (FusedBatchNormV3)
op: model/513/Conv2D (Conv2D)
op: model/514/FusedBatchNormV3 (FusedBatchNormV3)
op: model/516/Conv2D (Conv2D)
op: model/517/FusedBatchNormV3 (FusedBatchNormV3)
op: model/524/Conv2D (Conv2D)
op: model/525/FusedBatchNormV3 (FusedBatchNormV3)
op: model/527/depthwise (DepthwiseConv2dNative)
op: model/528/FusedBatchNormV3 (FusedBatchNormV3)
op: model/530/Conv2D (Conv2D)
op: model/531/FusedBatchNormV3 (FusedBatchNormV3)
op: model/533/Conv2D (Conv2D)
op: model/534/FusedBatchNormV3 (FusedBatchNormV3)
op: model/536/Conv2D (Conv2D)
op: model/537/FusedBatchNormV3 (FusedBatchNormV3)
op: model/545/Conv2D (Conv2D)
op: model/546/FusedBatchNormV3 (FusedBatchNormV3)
op: model/548/depthwise (DepthwiseConv2dNative)
op: model/549/FusedBatchNormV3 (FusedBatchNormV3)
op: model/551/Conv2D (Conv2D)
op: model/552/FusedBatchNormV3 (FusedBatchNormV3)
op: model/554/Conv2D (Conv2D)
op: model/555/FusedBatchNormV3 (FusedBatchNormV3)
op: model/557/Conv2D (Conv2D)
op: model/558/FusedBatchNormV3 (FusedBatchNormV3)
op: model/566/Conv2D (Conv2D)
op: model/567/FusedBatchNormV3 (FusedBatchNormV3)
op: model/574/depthwise (DepthwiseConv2dNative)
op: model/575/FusedBatchNormV3 (FusedBatchNormV3)
op: model/582/Conv2D (Conv2D)
op: model/583/FusedBatchNormV3 (FusedBatchNormV3)
op: model/584/Conv2D (Conv2D)
op: model/585/FusedBatchNormV3 (FusedBatchNormV3)
op: model/592/depthwise (DepthwiseConv2dNative)
op: model/593/FusedBatchNormV3 (FusedBatchNormV3)
op: model/600/Conv2D (Conv2D)
op: model/601/FusedBatchNormV3 (FusedBatchNormV3)
op: model/603/Conv2D (Conv2D)
op: model/604/FusedBatchNormV3 (FusedBatchNormV3)
op: model/611/depthwise (DepthwiseConv2dNative)
op: model/612/FusedBatchNormV3 (FusedBatchNormV3)
op: model/619/Conv2D (Conv2D)
op: model/620/FusedBatchNormV3 (FusedBatchNormV3)
op: model/622/Conv2D (Conv2D)
op: model/623/FusedBatchNormV3 (FusedBatchNormV3)
op: model/630/depthwise (DepthwiseConv2dNative)
op: model/631/FusedBatchNormV3 (FusedBatchNormV3)
op: model/638/Conv2D (Conv2D)
op: model/639/FusedBatchNormV3 (FusedBatchNormV3)
op: model/641/Conv2D (Conv2D)
op: model/642/FusedBatchNormV3 (FusedBatchNormV3)
op: model/649/depthwise (DepthwiseConv2dNative)
op: model/650/FusedBatchNormV3 (FusedBatchNormV3)
op: model/657/Conv2D (Conv2D)
op: model/658/FusedBatchNormV3 (FusedBatchNormV3)
op: model/660/Conv2D (Conv2D)
op: model/661/FusedBatchNormV3 (FusedBatchNormV3)
op: model/663/Conv2D (Conv2D)
op: model/664/FusedBatchNormV3 (FusedBatchNormV3)
op: model/671/Conv2D (Conv2D)
op: model/672/FusedBatchNormV3 (FusedBatchNormV3)
op: model/674/Conv2D (Conv2D)
op: model/675/FusedBatchNormV3 (FusedBatchNormV3)
op: model/682/depthwise (DepthwiseConv2dNative)
op: model/683/FusedBatchNormV3 (FusedBatchNormV3)
op: model/690/Conv2D (Conv2D)
op: model/691/FusedBatchNormV3 (FusedBatchNormV3)
op: model/693/Conv2D (Conv2D)
op: model/694/FusedBatchNormV3 (FusedBatchNormV3)
op: model/696/Conv2D (Conv2D)
op: model/697/FusedBatchNormV3 (FusedBatchNormV3)
op: model/705/Conv2D (Conv2D)
op: model/706/FusedBatchNormV3 (FusedBatchNormV3)
op: model/713/depthwise (DepthwiseConv2dNative)
op: model/714/FusedBatchNormV3 (FusedBatchNormV3)
op: model/721/Conv2D (Conv2D)
op: model/722/FusedBatchNormV3 (FusedBatchNormV3)
op: model/724/Conv2D (Conv2D)
op: model/725/FusedBatchNormV3 (FusedBatchNormV3)
op: model/727/Conv2D (Conv2D)
op: model/728/FusedBatchNormV3 (FusedBatchNormV3)
op: model/735/Conv2D (Conv2D)
op: model/736/FusedBatchNormV3 (FusedBatchNormV3)
op: model/738/Conv2D (Conv2D)
op: model/739/FusedBatchNormV3 (FusedBatchNormV3)
op: model/746/depthwise (DepthwiseConv2dNative)
op: model/747/FusedBatchNormV3 (FusedBatchNormV3)
op: model/754/Conv2D (Conv2D)
op: model/755/FusedBatchNormV3 (FusedBatchNormV3)
op: model/757/Conv2D (Conv2D)
op: model/758/FusedBatchNormV3 (FusedBatchNormV3)
op: model/760/Conv2D (Conv2D)
op: model/761/FusedBatchNormV3 (FusedBatchNormV3)
op: model/768/Conv2D (Conv2D)
op: model/769/FusedBatchNormV3 (FusedBatchNormV3)
op: model/776/depthwise (DepthwiseConv2dNative)
op: model/777/FusedBatchNormV3 (FusedBatchNormV3)
op: model/784/Conv2D (Conv2D)
op: model/785/FusedBatchNormV3 (FusedBatchNormV3)
op: model/787/Conv2D (Conv2D)
op: model/788/FusedBatchNormV3 (FusedBatchNormV3)
op: model/790/Conv2D (Conv2D)
op: model/791/FusedBatchNormV3 (FusedBatchNormV3)
op: model/799/Conv2D (Conv2D)
op: model/800/FusedBatchNormV3 (FusedBatchNormV3)
op: model/807/Conv2D (Conv2D)
op: model/808/FusedBatchNormV3 (FusedBatchNormV3)
op: model/815/Conv2D (Conv2D)
op: model/816/FusedBatchNormV3 (FusedBatchNormV3)
op: model/850/Conv2D (Conv2D)
op: model/851/FusedBatchNormV3 (FusedBatchNormV3)
op: model/858/Conv2D (Conv2D)
op: model/859/FusedBatchNormV3 (FusedBatchNormV3)
op: model/894/Conv2D (Conv2D)
op: model/895/FusedBatchNormV3 (FusedBatchNormV3)
op: model/902/Conv2D (Conv2D)
op: model/903/FusedBatchNormV3 (FusedBatchNormV3)
op: model/938/Conv2D (Conv2D)
op: model/939/FusedBatchNormV3 (FusedBatchNormV3)
op: model/946/Conv2D (Conv2D)
op: model/947/FusedBatchNormV3 (FusedBatchNormV3)
op: model/955/Conv2D (Conv2D)
op: model/956/FusedBatchNormV3 (FusedBatchNormV3)
op: model/963/Conv2D (Conv2D)
op: model/964/FusedBatchNormV3 (FusedBatchNormV3)
op: model/973/Conv2D (Conv2D)
op: model/974/FusedBatchNormV3 (FusedBatchNormV3)
op: model/981/Conv2D (Conv2D)
op: model/982/FusedBatchNormV3 (FusedBatchNormV3)
op: model/989/Conv2D (Conv2D)
op: model/990/FusedBatchNormV3 (FusedBatchNormV3)
op: model/999/Conv2D (Conv2D)
op: model/999/BiasAdd (BiasAdd)
op: model/1000/Conv2D (Conv2D)
op: model/1000/BiasAdd (BiasAdd)
op: model/landmark/Conv2D (Conv2D)
op: model/landmark/BiasAdd (BiasAdd)
"""