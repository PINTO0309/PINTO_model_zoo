import onnx
import onnx_graphsurgeon as gs
import sclblonnx as so

H=320
W=1024
MODEL1=f'model18_encoder_{H}x{W}.onnx'
MODEL2=f'model18_depth_decoder_{H}x{W}.onnx'
MODEL3=f'model18_encoder_depth_decoder_{H}x{W}.onnx'

graph1 = gs.import_onnx(onnx.load(MODEL1))
for n in graph1.nodes:
    for cn in n.inputs:
        if cn.name[-1:] != 'a':
            cn.name = f'{cn.name}a'
        else:
            pass
    for cn in n.outputs:
        if cn.name[-1:] != 'a':
            cn.name = f'{cn.name}a'
        else:
            pass
graph1_outputs = [o.name for o in graph1.outputs]
print(f'graph1 outputs: {graph1_outputs}')
onnx.save(gs.export_onnx(graph1), "graph1.onnx")

graph2 = gs.import_onnx(onnx.load(MODEL2))
graph2_inputs = []
for n in graph2.nodes:
    for cn in n.inputs:
        if cn.name[-1:] != 'b':
            cn.name = f'{cn.name}b'
        else:
            pass
    for cn in n.outputs:
        if cn.name[-1:] != 'b':
            cn.name = f'{cn.name}b'
        else:
            pass
graph2_inputs = [i.name for i in graph2.inputs]
print(f'graph2 inputs: {graph2_inputs}')
onnx.save(gs.export_onnx(graph2), "graph2.onnx")

sg1 = so.graph_from_file('graph1.onnx')
sg2 = so.graph_from_file('graph2.onnx')
sg3 = so.merge(
    sg1,
    sg2,
    outputs=graph1_outputs,
    inputs=graph2_inputs
)

so.graph_to_file(sg3, MODEL3)