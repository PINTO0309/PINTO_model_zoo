import os
import json
import base64
import numpy as np
from collections import namedtuple

import tensorflow as tf
import tensorflowjs as tfjs
import tensorflowjs.converters.common as tfjs_common
from tensorflowjs.read_weights import read_weights
from google.protobuf.json_format import ParseDict, MessageToDict

TFJS_NODE_KEY = 'node'

TFJS_NODE_ATTR_KEY = 'attr'
TFJS_NODE_CONST_KEY = 'Const'
TFJS_NODE_PLACEHOLDER_KEY = 'Placeholder'

TFJS_ATTR_DTYPE_KEY = 'dtype'
TFJS_ATTR_SHAPE_KEY = 'shape'
TFJS_ATTR_VALUE_KEY = 'value'
TFJS_ATTR_STRING_VALUE_KEY = 's'
TFJS_ATTR_INT_VALUE_KEY = 'i'


TFJS_NAME_KEY = 'name'
TFJS_DATA_KEY = 'data'


def _parse_path_and_model_json(model_dir):
    """
    Parse model directory name and return path and file name
    Args:
        model_dir: Model file path - either directory name or path + file name

    Returns:
        Tuple of directory name and model file name (without directory)
    """
    if model_dir.endswith('.json'):
        if not os.path.isfile(model_dir):
            raise ValueError("Model not found: {}".format(model_dir))
        return os.path.split(model_dir)
    elif os.path.isdir(model_dir):
        return model_dir, tfjs_common.ARTIFACT_MODEL_JSON_FILE_NAME
    else:
        raise ValueError("Model path is not a directory: {}".format(model_dir))


def _find_if_has_key(obj, key, of_type=None):
    """
    Recursively find all objects with a given key in a dictionary
    Args:
        obj: Dictionary to search
        key: Key to find
        of_type: [optional] Type of the referenced item

    Returns:
        List of all objects that contain an item with the given key and matching type
    """
    def children(item): return [
        val for val in item.values() if isinstance(val, dict)]
    found = []
    stack = children(obj)
    while len(stack) > 0:
        item = stack.pop()
        if key in item and (of_type is None or isinstance(item[key], of_type)):
            found.append(item)
        stack.extend(children(item))

    return found


def _convert_string_attrs(node):
    """
    Deep search string attributes (labelled "s" in GraphDef proto)
    and convert ascii code lists to base64-encoded strings if necessary
    """
    attr_key = TFJS_NODE_ATTR_KEY
    str_key = TFJS_ATTR_STRING_VALUE_KEY
    attrs = _find_if_has_key(node[attr_key], key=str_key, of_type=list)
    for attr in attrs:
        array = attr[str_key]
        # check if conversion is actually necessary
        if len(array) > 0 and isinstance(array, list) and isinstance(array[0], int):
            string = ''.join(map(chr, array))
            binary = string.encode('utf8')
            attr[str_key] = base64.encodebytes(binary)
        elif len(array) == 0:
            attr[str_key] = None

    return


def _fix_dilation_attrs(node):
    """
    Search dilations-attribute and convert
    misaligned dilation rates if necessary see
    https://github.com/patlevin/tfjs-to-tf/issues/1
    """
    path = ['attr', 'dilations', 'list']
    values = node
    for key in path:
        if key in values:
            values = values[key]
        else:
            values = None
            break

    # if dilations are present, they're stored in 'values' now
    ints = TFJS_ATTR_INT_VALUE_KEY
    if values is not None and ints in values and isinstance(values[ints], list):
        v = values[ints]
        if len(v) != 4:
            # must be NCHW-formatted 4D tensor or else TF can't handle it
            raise ValueError(
                "Unsupported 'dilations'-attribute in node {}".format(node[
                    TFJS_NAME_KEY]))
        # check for [>1,>1,1,1], which is likely a mistranslated [1,>1,>1,1]
        if int(v[0], 10) > 1:
            values[ints] = ['1', v[0], v[1], '1']

    return


def _convert_attr_values(message_dict):
    """
    Node attributes in deserialised JSON contain strings as lists of ascii codes.
    The TF GraphDef proto expects these values to be base64 encoded so convert all
    strings here.
    """
    if TFJS_NODE_KEY in message_dict:
        nodes = message_dict[TFJS_NODE_KEY]
        for node in nodes:
            _convert_string_attrs(node)
            _fix_dilation_attrs(node)

    return message_dict


def _convert_graph_def(message_dict):
    """
    Convert JSON to TF GraphDef message
    Args:
        message_dict: deserialised JSON message

    Returns:
        TF GraphDef message
    """
    message_dict = _convert_attr_values(message_dict)
    return ParseDict(message_dict, tf.compat.v1.GraphDef())


def _convert_weight_list_to_dict(weight_list):
    """
    Convert list of weight entries to dictionary
    Args:
        weight_list: List of numpy arrays or tensors formatted as
                     {'name': 'entry0', 'data': np.array([1,2,3], 'float32')}
    Returns:
        Dictionary that maps weight names to tensor data, e.g.
        {'entry0:': np.array(...), 'entry1': np.array(...), ...}
    """
    weight_dict = {}
    for entry in weight_list:
        weight_dict[entry[TFJS_NAME_KEY]] = entry[TFJS_DATA_KEY]
    return weight_dict


def _create_graph(graph_def, weight_dict):
    """
    Create a TF Graph from nodes
    Args:
        graph_def: TF GraphDef message containing the node graph
        weight_dict: Dictionary from node names to tensor data
    Returns:
        TF Graph for inference or saving
    """
    graph = tf.Graph()
    with tf.compat.v1.Session(graph=graph):
        for k, v in weight_dict.items():
            weight_dict[k] = tf.convert_to_tensor(v)
        tf.graph_util.import_graph_def(graph_def, weight_dict, name='')

    return graph


def _convert_graph_model_to_graph(model_json, base_path):
    """
    Convert TFJS JSON model to TF Graph
    Args:
        model_json: JSON dict from TFJS model file
        base_path:  Path to the model file (where to find the model weights)
    Returns:
        TF Graph for inference or saving
    """
    if not tfjs_common.ARTIFACT_MODEL_TOPOLOGY_KEY in model_json:
        raise ValueError("model_json is missing key '{}'".format(
            tfjs_common.ARTIFACT_MODEL_TOPOLOGY_KEY))

    topology = model_json[tfjs_common.ARTIFACT_MODEL_TOPOLOGY_KEY]

    if not tfjs_common.ARTIFACT_WEIGHTS_MANIFEST_KEY in model_json:
        raise ValueError("model_json is missing key '{}'".format(
            tfjs_common.ARTIFACT_WEIGHTS_MANIFEST_KEY))

    weights_manifest = model_json[tfjs_common.ARTIFACT_WEIGHTS_MANIFEST_KEY]
    weight_list = read_weights(weights_manifest, base_path, flatten=True)

    graph_def = _convert_graph_def(topology)
    weight_dict = _convert_weight_list_to_dict(weight_list)

    return _create_graph(graph_def, weight_dict)


def load_graph_model(model_dir):
    """
    Load a TFJS Graph Model from a directory
    Args:
        model_dir: Directory that contains the tfjs model.json and weights;
                alternatively name and path of the model.json if the name
                differs from the default ("model.json")
    Returns:
        TF frozen graph for inference or saving
    """
    model_path, model_name = _parse_path_and_model_json(model_dir)
    model_file_path = os.path.join(model_path, model_name)
    with open(model_file_path, "r") as f:
        model_json = json.load(f)
    return _convert_graph_model_to_graph(model_json, model_path)


_DTYPE_MAP = [
    None,
    np.float32,
    np.float64,
    np.int32,
    np.uint8,
    np.int16,
    np.int8,
    None,
    np.complex64,
    np.int64,
    np.bool
]

NodeInfo = namedtuple('NodeInfo', 'name shape dtype tensor')


def _is_op_node(node):
    return node.op not in (TFJS_NODE_CONST_KEY, TFJS_NODE_PLACEHOLDER_KEY)


def _op_nodes(graph_def):
    return [node for node in graph_def.node if _is_op_node(node)]


def _map_type(type_id):
    if type_id < 0 or type_id > len(_DTYPE_MAP):
        raise ValueError("Unsupported data type: {}".format(type_id))
    np_type = _DTYPE_MAP[type_id]
    return np_type


def _get_shape(node):
    def shape(attr): return attr.shape.dim
    def size(dim): return dim.size if dim.size > 0 else None
    return [size(dim) for dim in shape(node.attr[TFJS_ATTR_SHAPE_KEY])]


def _node_info(node):
    def dtype(n): return _map_type(n.attr[TFJS_ATTR_DTYPE_KEY].type)
    return NodeInfo(name=node.name, shape=_get_shape(node), dtype=dtype(node),
                    tensor=node.name + ':0')


def get_input_nodes(graph):
    """
    Return information about a graph's inputs.
    Arguments:
        graph: Graph or GraphDef object
    Returns:
        List of NodeInfo objects holding name, shape, and type of the input
    """
    if isinstance(graph, tf.Graph):
        graph_def = graph.as_graph_def()
    else:
        graph_def = graph
    nodes = [n for n in graph_def.node if n.op in (
        TFJS_NODE_PLACEHOLDER_KEY)]
    return [_node_info(node) for node in nodes]


def get_output_nodes(graph):
    """
    Return information about a graph's outputs.
    Arguments:
        graph: Graph or GraphDef object
    Returns:
        List of NodeInfo objects holding name, shape, and type of the input;
        shape will be left empty
    """
    if isinstance(graph, tf.Graph):
        graph_def = graph.as_graph_def()
    else:
        graph_def = graph

    ops = _op_nodes(graph_def)
    outputs = []
    for i in range(0, len(ops)):
        node = ops[i]
        has_ref = False
        for test in ops[i+1:]:
            if node.name in test.input:
                has_ref = True
                break
        if not has_ref:
            outputs.append(node)

    return [_node_info(node) for node in outputs]


def get_input_tensors(graph):
    """
    Return the names of the graph's input tensors.
    Arguments:
        graph: Graph or GraphDef object
    Returns:
        List of tensor names
    """
    return [node.tensor for node in get_input_nodes(graph)]


def get_output_tensors(graph):
    """
    Return the names of the graph's output tensors.
    Arguments:
        graph: Graph or GraphDef object
    Returns:
        List of tensor names
    """
    return [node.tensor for node in get_output_nodes(graph)]
