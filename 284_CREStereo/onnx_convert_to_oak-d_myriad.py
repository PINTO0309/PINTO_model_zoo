"""
Blog that verified the conversion
https://zenn.dev/pinto0309/scraps/475e4f2a641d22

Based ONNX file
https://github.com/PINTO0309/PINTO_model_zoo/blob/main/284_CREStereo/download_iter02_tensorrt.sh
"""

import os
import onnx
import onnx_graphsurgeon as gs
import numpy as np

# Load
onnx_file = 'crestereo_init_iter2_480x640.onnx'
onnx_graph = onnx.load(onnx_file)

# mul_shape = 70 #120x160
# mul_shape = 150 #160x240
# mul_shape = 220 #180x320
# mul_shape = 300 #240x320
# mul_shape = 600 #320x480
# mul_shape = 880 #360x640
mul_shape = 1200 #480x640
# mul_shape = 3600 #720x1280

graph = gs.import_onnx(onnx_graph)

# MVN
mod_sub_ops = {
    'Sub_650': 'onnx::Pow_1034',
    'Sub_504': 'onnx::Pow_693',
    'Sub_519': 'onnx::Pow_710',
    'Sub_451': 'onnx::Pow_603',
    'Sub_466': 'onnx::Pow_620',
    'Sub_595': 'onnx::Pow_942',
    'Sub_610': 'onnx::Pow_959',
    'Sub_665': 'onnx::Pow_1051',
}
mod_div_ops = {
    'Div_657': 'onnx::Pow_1034',
    'Div_511': 'onnx::Pow_693',
    'Div_526': 'onnx::Pow_710',
    'Div_458': 'onnx::Pow_603',
    'Div_473': 'onnx::Pow_620',
    'Div_602': 'onnx::Pow_942',
    'Div_617': 'onnx::Pow_959',
    'Div_672': 'onnx::Pow_1051',
}

# Less
mod_add_ops = {
    'Add_1542': 'onnx::Less_2218',
    'Add_1538': 'onnx::Less_2214',
    'Add_1544': 'onnx::Less_2220',
    'Add_1052': 'onnx::Less_1570',
    'Add_1054': 'onnx::Less_1572',
    'Add_1048': 'onnx::Less_1566',
    'Add_1540': 'onnx::Less_2216',
    'Add_805': 'onnx::Less_1244',
    'Add_807': 'onnx::Less_1246',
    'Add_809': 'onnx::Less_1248',
    'Add_803': 'onnx::Less_1242',
    'Add_1050': 'onnx::Less_1568',
    'Add_1295': 'onnx::Less_1892',
    'Add_1297': 'onnx::Less_1894',
    'Add_1299': 'onnx::Less_1896',
    'Add_1293': 'onnx::Less_1890',
    'Add_2664': 'onnx::Less_3705',
    'Add_2670': 'onnx::Less_3711',
    'Add_2668': 'onnx::Less_3709',
    'Add_2666': 'onnx::Less_3707',
    'Add_2420': 'onnx::Less_3382',
    'Add_2426': 'onnx::Less_3388',
    'Add_2424': 'onnx::Less_3386',
    'Add_2422': 'onnx::Less_3384',
    'Add_2176': 'onnx::Less_3059',
    'Add_2182': 'onnx::Less_3065',
    'Add_2180': 'onnx::Less_3063',
    'Add_2178': 'onnx::Less_3061',
    'Add_1932': 'onnx::Less_2736',
    'Add_1938': 'onnx::Less_2742',
    'Add_1936': 'onnx::Less_2740',
    'Add_1934': 'onnx::Less_2738',
    'Add_3022': 'onnx::Less_4166',
    'Add_3028': 'onnx::Less_4172',
    'Add_3026': 'onnx::Less_4170',
    'Add_3024': 'onnx::Less_4168',
    'Add_3742': 'onnx::Less_4991',
    'Add_3748': 'onnx::Less_4997',
    'Add_3746': 'onnx::Less_4995',
    'Add_3744': 'onnx::Less_4993',
}
mod_less_ops = {
    'Less_1562': 'onnx::Less_2218',
    'Less_1546': 'onnx::Less_2214',
    'Less_1570': 'onnx::Less_2220',
    'Less_1072': 'onnx::Less_1570',
    'Less_1080': 'onnx::Less_1572',
    'Less_1056': 'onnx::Less_1566',
    'Less_1554': 'onnx::Less_2216',
    'Less_819': 'onnx::Less_1244',
    'Less_827': 'onnx::Less_1246',
    'Less_835': 'onnx::Less_1248',
    'Less_811': 'onnx::Less_1242',
    'Less_1064': 'onnx::Less_1568',
    'Less_1309': 'onnx::Less_1892',
    'Less_1317': 'onnx::Less_1894',
    'Less_1325': 'onnx::Less_1896',
    'Less_1301': 'onnx::Less_1890',
    'Less_2672': 'onnx::Less_3705',
    'Less_2696': 'onnx::Less_3711',
    'Less_2688': 'onnx::Less_3709',
    'Less_2680': 'onnx::Less_3707',
    'Less_2428': 'onnx::Less_3382',
    'Less_2452': 'onnx::Less_3388',
    'Less_2444': 'onnx::Less_3386',
    'Less_2436': 'onnx::Less_3384',
    'Less_2184': 'onnx::Less_3059',
    'Less_2208': 'onnx::Less_3065',
    'Less_2200': 'onnx::Less_3063',
    'Less_2192': 'onnx::Less_3061',
    'Less_1940': 'onnx::Less_2736',
    'Less_1964': 'onnx::Less_2742',
    'Less_1956': 'onnx::Less_2740',
    'Less_1948': 'onnx::Less_2738',
    'Less_3030': 'onnx::Less_4166',
    'Less_3054': 'onnx::Less_4172',
    'Less_3046': 'onnx::Less_4170',
    'Less_3038': 'onnx::Less_4168',
    'Less_3750': 'onnx::Less_4991',
    'Less_3774': 'onnx::Less_4997',
    'Less_3766': 'onnx::Less_4995',
    'Less_3758': 'onnx::Less_4993',
}

# Greater
mod_where_ops = {
    'Where_1564': 'onnx::Greater_2240',
    'Where_1548': 'onnx::Greater_2224',
    'Where_1572': 'onnx::Greater_2248',
    'Where_1074': 'onnx::Greater_1592',
    'Where_1082': 'onnx::Greater_1600',
    'Where_1058': 'onnx::Greater_1576',
    'Where_1556': 'onnx::Greater_2232',
    'Where_821': 'onnx::Greater_1260',
    'Where_829': 'onnx::Greater_1268',
    'Where_837': 'onnx::Greater_1276',
    'Where_813': 'onnx::Greater_1252',
    'Where_1066': 'onnx::Greater_1584',
    'Where_1311': 'onnx::Greater_1908',
    'Where_1319': 'onnx::Greater_1916',
    'Where_1327': 'onnx::Greater_1924',
    'Where_1303': 'onnx::Greater_1900',
    'Where_2674': 'onnx::Greater_3715',
    'Where_2698': 'onnx::Greater_3739',
    'Where_2690': 'onnx::Greater_3731',
    'Where_2682': 'onnx::Greater_3723',
    'Where_2430': 'onnx::Greater_3392',
    'Where_2454': 'onnx::Greater_3416',
    'Where_2446': 'onnx::Greater_3408',
    'Where_2438': 'onnx::Greater_3400',
    'Where_2186': 'onnx::Greater_3069',
    'Where_2210': 'onnx::Greater_3093',
    'Where_2202': 'onnx::Greater_3085',
    'Where_2194': 'onnx::Greater_3077',
    'Where_1942': 'onnx::Greater_2746',
    'Where_1966': 'onnx::Greater_2770',
    'Where_1958': 'onnx::Greater_2762',
    'Where_1950': 'onnx::Greater_2754',
    'Where_3032': 'onnx::Greater_4176',
    'Where_3056': 'onnx::Greater_4200',
    'Where_3048': 'onnx::Greater_4192',
    'Where_3040': 'onnx::Greater_4184',
    'Where_3752': 'onnx::Greater_5001',
    'Where_3776': 'onnx::Greater_5025',
    'Where_3768': 'onnx::Greater_5017',
    'Where_3760': 'onnx::Greater_5009',
}
mod_greater_ops = {
    'Greater_1566': 'onnx::Greater_2240',
    'Greater_1550': 'onnx::Greater_2224',
    'Greater_1574': 'onnx::Greater_2248',
    'Greater_1076': 'onnx::Greater_1592',
    'Greater_1084': 'onnx::Greater_1600',
    'Greater_1060': 'onnx::Greater_1576',
    'Greater_1558': 'onnx::Greater_2232',
    'Greater_823': 'onnx::Greater_1260',
    'Greater_831': 'onnx::Greater_1268',
    'Greater_839': 'onnx::Greater_1276',
    'Greater_815': 'onnx::Greater_1252',
    'Greater_1068': 'onnx::Greater_1584',
    'Greater_1313': 'onnx::Greater_1908',
    'Greater_1321': 'onnx::Greater_1916',
    'Greater_1329': 'onnx::Greater_1924',
    'Greater_1305': 'onnx::Greater_1900',
    'Greater_2676': 'onnx::Greater_3715',
    'Greater_2700': 'onnx::Greater_3739',
    'Greater_2692': 'onnx::Greater_3731',
    'Greater_2684': 'onnx::Greater_3723',
    'Greater_2432': 'onnx::Greater_3392',
    'Greater_2456': 'onnx::Greater_3416',
    'Greater_2448': 'onnx::Greater_3408',
    'Greater_2440': 'onnx::Greater_3400',
    'Greater_2188': 'onnx::Greater_3069',
    'Greater_2212': 'onnx::Greater_3093',
    'Greater_2204': 'onnx::Greater_3085',
    'Greater_2196': 'onnx::Greater_3077',
    'Greater_1944': 'onnx::Greater_2746',
    'Greater_1968': 'onnx::Greater_2770',
    'Greater_1960': 'onnx::Greater_2762',
    'Greater_1952': 'onnx::Greater_2754',
    'Greater_3034': 'onnx::Greater_4176',
    'Greater_3058': 'onnx::Greater_4200',
    'Greater_3050': 'onnx::Greater_4192',
    'Greater_3042': 'onnx::Greater_4184',
    'Greater_3754': 'onnx::Greater_5001',
    'Greater_3778': 'onnx::Greater_5025',
    'Greater_3770': 'onnx::Greater_5017',
    'Greater_3762': 'onnx::Greater_5009',
}

# Expand
mod_unsqueeze_ops = {
    'Unsqueeze_1591': 'onnx::Expand_2281',
    'Unsqueeze_1581': 'onnx::Expand_2262',
    'Unsqueeze_1601': 'onnx::Expand_2300',
    'Unsqueeze_1611': 'onnx::Expand_2319',
    'Unsqueeze_1346': 'onnx::Expand_1957',
    'Unsqueeze_1336': 'onnx::Expand_1938',
    'Unsqueeze_1356': 'onnx::Expand_1976',
    'Unsqueeze_1366': 'onnx::Expand_1995',
    'Unsqueeze_1101': 'onnx::Expand_1633',
    'Unsqueeze_1091': 'onnx::Expand_1614',
    'Unsqueeze_1111': 'onnx::Expand_1652',
    'Unsqueeze_1121': 'onnx::Expand_1671',
    'Unsqueeze_856': 'onnx::Expand_1309',
    'Unsqueeze_846': 'onnx::Expand_1290',
    'Unsqueeze_866': 'onnx::Expand_1328',
    'Unsqueeze_876': 'onnx::Expand_1347',
    'Unsqueeze_2717': 'onnx::Expand_3772',
    'Unsqueeze_2707': 'onnx::Expand_3753',
    'Unsqueeze_2727': 'onnx::Expand_3791',
    'Unsqueeze_2737': 'onnx::Expand_3810',
    'Unsqueeze_2473': 'onnx::Expand_3449',
    'Unsqueeze_2463': 'onnx::Expand_3430',
    'Unsqueeze_2483': 'onnx::Expand_3468',
    'Unsqueeze_2493': 'onnx::Expand_3487',
    'Unsqueeze_2229': 'onnx::Expand_3126',
    'Unsqueeze_2219': 'onnx::Expand_3107',
    'Unsqueeze_2239': 'onnx::Expand_3145',
    'Unsqueeze_2249': 'onnx::Expand_3164',
    'Unsqueeze_1985': 'onnx::Expand_2803',
    'Unsqueeze_1975': 'onnx::Expand_2784',
    'Unsqueeze_1995': 'onnx::Expand_2822',
    'Unsqueeze_2005': 'onnx::Expand_2841',
    'Unsqueeze_3075': 'onnx::Expand_4233',
    'Unsqueeze_3065': 'onnx::Expand_4214',
    'Unsqueeze_3085': 'onnx::Expand_4252',
    'Unsqueeze_3095': 'onnx::Expand_4271',
    'Unsqueeze_3795': 'onnx::Expand_5058',
    'Unsqueeze_3785': 'onnx::Expand_5039',
    'Unsqueeze_3805': 'onnx::Expand_5077',
    'Unsqueeze_3815': 'onnx::Expand_5096',
}
mod_expand_ops = {
    'Expand_1597': 'onnx::Expand_2281',
    'Expand_1587': 'onnx::Expand_2262',
    'Expand_1607': 'onnx::Expand_2300',
    'Expand_1617': 'onnx::Expand_2319',
    'Expand_1352': 'onnx::Expand_1957',
    'Expand_1342': 'onnx::Expand_1938',
    'Expand_1362': 'onnx::Expand_1976',
    'Expand_1372': 'onnx::Expand_1995',
    'Expand_1107': 'onnx::Expand_1633',
    'Expand_1097': 'onnx::Expand_1614',
    'Expand_1117': 'onnx::Expand_1652',
    'Expand_1127': 'onnx::Expand_1671',
    'Expand_862': 'onnx::Expand_1309',
    'Expand_852': 'onnx::Expand_1290',
    'Expand_872': 'onnx::Expand_1328',
    'Expand_882': 'onnx::Expand_1347',
    'Expand_2723': 'onnx::Expand_3772',
    'Expand_2713': 'onnx::Expand_3753',
    'Expand_2733': 'onnx::Expand_3791',
    'Expand_2743': 'onnx::Expand_3810',
    'Expand_2479': 'onnx::Expand_3449',
    'Expand_2469': 'onnx::Expand_3430',
    'Expand_2489': 'onnx::Expand_3468',
    'Expand_2499': 'onnx::Expand_3487',
    'Expand_2235': 'onnx::Expand_3126',
    'Expand_2225': 'onnx::Expand_3107',
    'Expand_2245': 'onnx::Expand_3145',
    'Expand_2255': 'onnx::Expand_3164',
    'Expand_1991': 'onnx::Expand_2803',
    'Expand_1981': 'onnx::Expand_2784',
    'Expand_2001': 'onnx::Expand_2822',
    'Expand_2011': 'onnx::Expand_2841',
    'Expand_3081': 'onnx::Expand_4233',
    'Expand_3071': 'onnx::Expand_4214',
    'Expand_3091': 'onnx::Expand_4252',
    'Expand_3101': 'onnx::Expand_4271',
    'Expand_3801': 'onnx::Expand_5058',
    'Expand_3791': 'onnx::Expand_5039',
    'Expand_3811': 'onnx::Expand_5077',
    'Expand_3821': 'onnx::Expand_5096',
}
mod_gatherelements_ops = {
    'GatherElements_1619': 'onnx::GatherElements_2296',
    'GatherElements_1618': 'onnx::GatherElements_2277',
    'GatherElements_1620': 'onnx::GatherElements_2315',
    'GatherElements_1621': 'onnx::GatherElements_2334',
    'GatherElements_1374': 'onnx::GatherElements_1972',
    'GatherElements_1373': 'onnx::GatherElements_1953',
    'GatherElements_1375': 'onnx::GatherElements_1991',
    'GatherElements_1376': 'onnx::GatherElements_2010',
    'GatherElements_1129': 'onnx::GatherElements_1648',
    'GatherElements_1128': 'onnx::GatherElements_1629',
    'GatherElements_1130': 'onnx::GatherElements_1667',
    'GatherElements_1131': 'onnx::GatherElements_1686',
    'GatherElements_884': 'onnx::GatherElements_1324',
    'GatherElements_883': 'onnx::GatherElements_1305',
    'GatherElements_885': 'onnx::GatherElements_1343',
    'GatherElements_886': 'onnx::GatherElements_1362',
    'GatherElements_2745': 'onnx::GatherElements_3787',
    'GatherElements_2744': 'onnx::GatherElements_3768',
    'GatherElements_2746': 'onnx::GatherElements_3806',
    'GatherElements_2747': 'onnx::GatherElements_3825',
    'GatherElements_2501': 'onnx::GatherElements_3464',
    'GatherElements_2500': 'onnx::GatherElements_3445',
    'GatherElements_2502': 'onnx::GatherElements_3483',
    'GatherElements_2503': 'onnx::GatherElements_3502',
    'GatherElements_2257': 'onnx::GatherElements_3141',
    'GatherElements_2256': 'onnx::GatherElements_3122',
    'GatherElements_2258': 'onnx::GatherElements_3160',
    'GatherElements_2259': 'onnx::GatherElements_3179',
    'GatherElements_2013': 'onnx::GatherElements_2818',
    'GatherElements_2012': 'onnx::GatherElements_2799',
    'GatherElements_2014': 'onnx::GatherElements_2837',
    'GatherElements_2015': 'onnx::GatherElements_2856',
    'GatherElements_3103': 'onnx::GatherElements_4248',
    'GatherElements_3102': 'onnx::GatherElements_4229',
    'GatherElements_3104': 'onnx::GatherElements_4267',
    'GatherElements_3105': 'onnx::GatherElements_4286',
    'GatherElements_3823': 'onnx::GatherElements_5073',
    'GatherElements_3822': 'onnx::GatherElements_5054',
    'GatherElements_3824': 'onnx::GatherElements_5092',
    'GatherElements_3825': 'onnx::GatherElements_5111',
}


#### MVN
for mod_sub_op, mod_sub_op_output, mod_div_op, mod_div_op_input in zip(mod_sub_ops.keys(), mod_sub_ops.values(), mod_div_ops.keys(), mod_div_ops.values()):

    # Finding outputs for connection
    mod_output = None
    for graph_node in graph.nodes:
        if graph_node.name == mod_sub_op:
            for node_output in graph_node.outputs:
                if node_output.name == mod_sub_op_output:
                    mod_output = node_output
                    break

    # Generate dummy Multiply OP
    # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Mul-7
    # https://docs.openvino.ai/latest/openvino_docs_ops_arithmetic_Multiply_1.html
    mul_node = gs.Node(
        op="Mul",
        name=f"{mod_output.name}_dummy_mul",
        inputs=[
            mod_output,
            gs.Constant(
                name=f"{mod_output.name}_dummy_mul_input",
                values=np.asarray([1.0], dtype=np.float32)
            ),
        ],
        outputs=[
            gs.Variable(
                name=f"{mod_output.name}_dummy_mul_output",
                dtype=np.float32,
                # shape=[1,70,256],
                shape=[1,mul_shape,256],
            ),
        ]
    )
    graph.nodes.append(mul_node)

    # Finding inputs for connection
    for node_idx, graph_node in enumerate(graph.nodes):
        if graph_node.name == mod_div_op:
            for input_idx, node_input in enumerate(graph_node.inputs):
                if node_input.name == mod_div_op_input:
                    graph.nodes[node_idx].inputs[input_idx] = mul_node.outputs[0]
                    break

#### Less
for mod_add_op, mod_add_op_output, mod_less_op, mod_less_op_input in zip(mod_add_ops.keys(), mod_add_ops.values(), mod_less_ops.keys(), mod_less_ops.values()):
    # Finding outputs for connection
    mod_output = None
    for graph_node in graph.nodes:
        if graph_node.name == mod_add_op:
            for node_output in graph_node.outputs:
                if node_output.name == mod_add_op_output:
                    mod_output = node_output
                    break

    # Generate dummy Cast OP
    # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Cast-9
    # https://docs.openvino.ai/latest/openvino_docs_ops_type_Convert_1.html
    cast_node = gs.Node(
        op="Cast",
        name=f"{mod_output.name}_dummy_cast",
        attrs={"to": onnx.TensorProto.FLOAT},
        inputs=[
            mod_output,
        ],
        outputs=[
            gs.Variable(
                name=f"{mod_output.name}_dummy_cast_output",
                dtype=np.float32,
                shape=mod_output.shape,
            ),
        ]
    )
    graph.nodes.append(cast_node)

    # Finding inputs for connection
    for node_idx, graph_node in enumerate(graph.nodes):
        if graph_node.name == mod_less_op:
            for input_idx, node_input in enumerate(graph_node.inputs):
                if node_input.name == mod_less_op_input:
                    graph.nodes[node_idx].inputs[input_idx] = cast_node.outputs[0]
                    break

#### Grater
for mod_where_op, mod_where_op_output, mod_greater_op, mod_greater_op_input in zip(mod_where_ops.keys(), mod_where_ops.values(), mod_greater_ops.keys(), mod_greater_ops.values()):
    # Finding outputs for connection
    mod_output = None
    for graph_node in graph.nodes:
        if graph_node.name == mod_where_op:
            for node_output in graph_node.outputs:
                if node_output.name == mod_where_op_output:
                    mod_output = node_output
                    break

    # Generate dummy Cast OP
    # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Cast-9
    # https://docs.openvino.ai/latest/openvino_docs_ops_type_Convert_1.html
    cast_node = gs.Node(
        op="Cast",
        name=f"{mod_output.name}_dummy_cast",
        attrs={"to": onnx.TensorProto.FLOAT},
        inputs=[
            mod_output,
        ],
        outputs=[
            gs.Variable(
                name=f"{mod_output.name}_dummy_cast_output",
                dtype=np.float32,
                shape=mod_output.shape,
            ),
        ]
    )
    graph.nodes.append(cast_node)

    # Finding inputs for connection
    for node_idx, graph_node in enumerate(graph.nodes):
        if graph_node.name == mod_greater_op:
            for input_idx, node_input in enumerate(graph_node.inputs):
                if node_input.name == mod_greater_op_input:
                    graph.nodes[node_idx].inputs[input_idx] = cast_node.outputs[0]
                    break

#### Expand
for mod_unsqueeze_op, mod_unsqueeze_op_output, mod_expand_op, mod_expand_op_input, mod_gatherelements_op, mod_gatherelements_op_input in zip(mod_unsqueeze_ops.keys(), mod_unsqueeze_ops.values(), mod_expand_ops.keys(), mod_expand_ops.values(), mod_gatherelements_ops.keys(), mod_gatherelements_ops.values()):
    # Finding outputs for connection
    mod_output = None
    for graph_node in graph.nodes:
        if graph_node.name == mod_unsqueeze_op:
            for node_output in graph_node.outputs:
                if node_output.name == mod_unsqueeze_op_output:
                    mod_output = node_output
                    break

    # Generate dummy Cast OP From
    # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Cast-9
    # https://docs.openvino.ai/latest/openvino_docs_ops_type_Convert_1.html
    cast_node_from = gs.Node(
        op="Cast",
        name=f"{mod_output.name}_dummy_cast_from",
        attrs={"to": onnx.TensorProto.FLOAT},
        inputs=[
            mod_output,
        ],
        outputs=[
            gs.Variable(
                name=f"{mod_output.name}_dummy_cast_from_output",
                dtype=np.float32,
                shape=mod_output.shape,
            ),
        ]
    )
    graph.nodes.append(cast_node_from)

    # Finding inputs for connection
    mod_input = None
    expand_op = None
    for node_idx, graph_node in enumerate(graph.nodes):
        if graph_node.name == mod_expand_op:
            for input_idx, node_input in enumerate(graph_node.inputs):
                if node_input.name == mod_expand_op_input:
                    graph.nodes[node_idx].inputs[input_idx] = cast_node_from.outputs[0]
                    graph.nodes[node_idx].outputs[0].dtype = np.float32
                    expand_op = graph_node
                    break

    # Generate dummy Cast OP To
    # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Cast-9
    # https://docs.openvino.ai/latest/openvino_docs_ops_type_Convert_1.html
    cast_node_to = gs.Node(
        op="Cast",
        name=f"{mod_output.name}_dummy_cast_to",
        attrs={"to": onnx.TensorProto.INT64},
        inputs=[
            expand_op.outputs[0],
        ],
        outputs=[
            gs.Variable(
                name=f"{mod_output.name}_dummy_cast_to_output",
                dtype=np.int64,
                shape=expand_op.outputs[0].shape,
            ),
        ]
    )
    graph.nodes.append(cast_node_to)

    # Finding inputs for connection
    for node_idx, graph_node in enumerate(graph.nodes):
        if graph_node.name == mod_gatherelements_op:
            for input_idx, node_input in enumerate(graph_node.inputs):
                if node_input.name == mod_gatherelements_op_input:
                    graph.nodes[node_idx].inputs[input_idx] = cast_node_to.outputs[0]
                    break

# Cleanup
graph.cleanup().toposort()
# Export
changed_graph = gs.export_onnx(graph)
# Shape inference
new_model = onnx.shape_inference.infer_shapes(changed_graph)
# Save
onnx.save(new_model, f'{os.path.splitext(onnx_file)[0]}_myriad_oak.onnx')
