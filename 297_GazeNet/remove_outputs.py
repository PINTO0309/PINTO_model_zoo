import onnx
import onnx_graphsurgeon as gs

MODEL='gazenet_gafa_NxFx3x256x192'

onnx_graph = onnx.load(f'{MODEL}.onnx')
graph = gs.import_onnx(onnx_graph)
graph.cleanup().toposort()


graph.outputs = [graph_output for idx, graph_output in enumerate(graph.outputs) if idx <= 2]

print(graph.outputs)

graph.cleanup().toposort()
optimized_graph = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
onnx.save(optimized_graph, f'{MODEL}_opt.onnx')
