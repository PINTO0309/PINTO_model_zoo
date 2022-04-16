import onnx
import onnx_graphsurgeon as gs

ORG_MODEL = 'onnx_net_merged/model_float32'
graph = gs.import_onnx(onnx.load(f'{ORG_MODEL}.onnx'))

# net1_Restoration_net/Pad
access_point_node = None
for node in graph.nodes:
    if node.name == 'net1_Restoration_net/Pad':
        access_point_node = node
        break

# net2_reconnet/add
for node in graph.nodes:
    if node.name == 'net2_reconnet/add':
        node.inputs[0] = access_point_node.outputs[0]
        break

graph.inputs.remove(graph.inputs[1])

graph.cleanup().toposort()
new_model = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
onnx.save(new_model, f'{ORG_MODEL}_final.onnx')