import onnx
import onnx_graphsurgeon as gs

input_onnx_file_paths = [
    'model_float32_camera.onnx',
    'model_float32_chair.onnx',
    'model_float32_cup.onnx',
    'model_float32_sneakers.onnx',
]

prefixes = [
    'camera',
    'chair',
    'cup',
    'sneakers',
]

src_onnx_graphs = []
for onnx_path in input_onnx_file_paths:
    gs_graph = gs.import_onnx(onnx.load(onnx_path))
    gs_graph.cleanup().toposort()
    src_onnx_graphs.append(gs.export_onnx(gs_graph))

pref_src_onnx_gs_graphs = []
for src_onnx_graph, prefixe in zip(src_onnx_graphs, prefixes):
    pref_src_onnx_gs_graphs.append(
        gs.import_onnx(onnx.compose.add_prefix(src_onnx_graph, prefix=f'{prefixe}_'))
    )

for idx in range(1, len(pref_src_onnx_gs_graphs)):

    for node_idx, node in enumerate(pref_src_onnx_gs_graphs[idx].nodes):
        if node_idx == 0:
            pref_src_onnx_gs_graphs[0].nodes.append(node)
            pref_src_onnx_gs_graphs[0].nodes[-1].inputs[0] = pref_src_onnx_gs_graphs[0].inputs[0]
        else:
            pref_src_onnx_gs_graphs[0].nodes.append(node)

    for output in pref_src_onnx_gs_graphs[idx].outputs:
        pref_src_onnx_gs_graphs[0].outputs.append(output)


pref_src_onnx_gs_graphs[0].inputs[0].name = "input"
pref_src_onnx_gs_graphs[0].cleanup().toposort()

merged_onnx_graph = gs.export_onnx(pref_src_onnx_gs_graphs[0])
onnx.save(merged_onnx_graph, "objectron_camera_chair_cup_sneakers.onnx")