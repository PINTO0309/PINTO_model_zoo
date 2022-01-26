import onnx
import os
import struct

from argparse import ArgumentParser

def rebatch(infile, outfile, batch_size):
    model = onnx.load(infile)
    graph = model.graph

    # Change batch size in input, output and value_info
    for tensor in list(graph.input) + list(graph.value_info) + list(graph.output):
        tensor.type.tensor_type.shape.dim[0].dim_param = batch_size

    # Set dynamic batch size in reshapes (-1)
    for node in  graph.node:
        if node.op_type != 'Reshape':
            continue
        for init in graph.initializer:
            # node.input[1] is expected to be a reshape
            if init.name != node.input[1]:
                continue
            # Shape is stored as a list of ints
            if len(init.int64_data) > 0:
                # This overwrites bias nodes' reshape shape but should be fine
                init.int64_data[0] = -1
            # Shape is stored as bytes
            elif len(init.raw_data) > 0:
                shape = bytearray(init.raw_data)
                struct.pack_into('q', shape, 0, -1)
                init.raw_data = bytes(shape)

    onnx.save(model, outfile)

if __name__ == '__main__':
    parser = ArgumentParser('Replace batch size with \'N\'')
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args = parser.parse_args()

    rebatch(args.infile, args.outfile, 'N')