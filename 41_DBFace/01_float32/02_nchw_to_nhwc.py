### tensorflow==2.2.0

import tensorflow as tf
if not tf.__version__.startswith('1'):
  import tensorflow.compat.v1 as tf
from tensorflow.python.tools import optimize_for_inference_lib

graph_def_file = "dbface.pb"

tf.reset_default_graph()
graph_def = tf.GraphDef()
with tf.Session() as sess:
    # Read binary pb graph from file
    with tf.gfile.Open(graph_def_file, "rb") as f:
        data2read = f.read()
        graph_def.ParseFromString(data2read)
    tf.graph_util.import_graph_def(graph_def, name='')
    
    # Get Nodes
    conv_nodes = [n for n in sess.graph.get_operations() if n.type in ['Conv2D','MaxPool','AvgPool','DepthwiseConv2dNative']]
    for n_org in conv_nodes:
        # Transpose input
        assert len(n_org.inputs)==1 or len(n_org.inputs)==2
        org_inp_tens = sess.graph.get_tensor_by_name(n_org.inputs[0].name)
        inp_tens = tf.transpose(org_inp_tens, [0, 2, 3, 1], name=n_org.name +'_transp_input')
        op_inputs = [inp_tens]
        
        # # Get filters for Conv but don't transpose
        # if n_org.type == 'Conv2D':
        #     filter_tens = sess.graph.get_tensor_by_name(n_org.inputs[1].name)
        #     op_inputs.append(filter_tens)
        # Get filters for Conv but don't transpose
        if n_org.type in ['Conv2D','DepthwiseConv2dNative']:
            filter_tens = sess.graph.get_tensor_by_name(n_org.inputs[1].name)
            op_inputs.append(filter_tens)
        
        # Attributes without data_format, NWHC is default
        atts = {key:n_org.node_def.attr[key] for key in list(n_org.node_def.attr.keys()) if key != 'data_format'}
        # if n_org.type in['MaxPool', 'AvgPool','Conv2D']:
        #     st = atts['strides'].list.i
        #     stl = [st[0], st[2], st[3], st[1]]
        #     atts['strides'] = tf.AttrValue(list=tf.AttrValue.ListValue(i=stl))
        if n_org.type in['MaxPool', 'AvgPool','Conv2D','DepthwiseConv2dNative']:
            st = atts['strides'].list.i
            stl = [st[0], st[2], st[3], st[1]]
            atts['strides'] = tf.AttrValue(list=tf.AttrValue.ListValue(i=stl))
        if n_org.type in ['MaxPool', 'AvgPool']:
            st = atts['ksize'].list.i
            stl = [st[0], st[2], st[3], st[1]]
            atts['ksize'] = tf.AttrValue(list=tf.AttrValue.ListValue(i=stl))

        # Create new Operation
        #print(n_org.type, n_org.name, list(n_org.inputs), n_org.node_def.attr['data_format'])
        op = sess.graph.create_op(op_type=n_org.type, inputs=op_inputs, name=n_org.name+'_new', dtypes=[tf.float32], attrs=atts) 
        out_tens = sess.graph.get_tensor_by_name(n_org.name+'_new'+':0')
        out_trans = tf.transpose(out_tens, [0, 3, 1, 2], name=n_org.name +'_transp_out')
        print('out_trans.shape:', out_trans.shape)
        print('sess.graph.get_tensor_by_name(n_org.name+\':0\').shape:', sess.graph.get_tensor_by_name(n_org.name+':0').shape)
        #assert out_trans.shape == sess.graph.get_tensor_by_name(n_org.name+':0').shape
        
        # Update Connections
        out_nodes = [n for n in sess.graph.get_operations() if n_org.outputs[0] in n.inputs]
        for out in out_nodes:
            for j, nam in enumerate(out.inputs):
                if n_org.outputs[0] == nam:
                    out._update_input(j, out_trans)
        
    # Delete old nodes
    graph_def = sess.graph.as_graph_def()
    for on in conv_nodes:
        graph_def.node.remove(on.node_def)

    # Write graph
    tf.io.write_graph(graph_def, "", graph_def_file.rsplit('.', 1)[0]+'_nhwc.pb', as_text=False)