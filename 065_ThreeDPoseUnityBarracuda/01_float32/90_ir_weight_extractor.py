### python3 ir_weight_extractor.py -m openvino/480x640/FP32/u2netp_480x640.xml -o weights_u2netp/

import os
import sys
import argparse
import pickle
import struct
import numpy as np

import xml.etree.ElementTree as et

from openvino.inference_engine import IECore

def dumpWeight(model, output_path):
    # for unpacking binary buffer
    format_config = { 'FP32': ['f', 4], 
                      'FP16': ['e', 2],
                      'I64' : ['q', 8],
                      'I32' : ['i', 4],
                      'I16' : ['h', 2],
                      'I8'  : ['b', 1],
                      'U8'  : ['B', 1]}

    # Read IR weight data
    with open(model+'.bin', 'rb') as f:
        binWeight = f.read()

    # Parse IR XML file, find 'Const' node, extract weight, and generate npy file 
    tree = et.parse(model+'.xml')
    root = tree.getroot()
    layers = root.find('layers')
    weight = {}
    print('size : nodeName')
    for layer in layers:
        if layer.attrib['type'] == 'Const':
            data = layer.find('data')
            if not data is None:
                if 'offset' in data.attrib and 'size' in data.attrib:
                    offset = int(data.attrib['offset'])
                    size   = int(data.attrib['size'])
                    shape_str  = data.attrib['shape'].split(',')
                    shape = [int(s) for s in shape_str]
                    blobBin = binWeight[offset:offset+size]                     # cut out the weight for this blob from the weight buffer
                    prec = layer.find('output').find('port').attrib['precision']
                    formatstring = '<' + format_config[prec][0] * (len(blobBin)//format_config[prec][1])
                    decodedwgt = np.array(list(struct.unpack(formatstring, blobBin))).reshape(shape)           # decode the buffer
                    weight[layer.attrib['name']] = [ prec, decodedwgt ]         # { blobName : [ precStr, weightBuf ]}
                    print('{} : {}'.format(len(blobBin), layer.attrib['name']))

                    print(layer.attrib['name'].replace('/', '_'), shape)
                    print('decodedwgt.shape:', decodedwgt.shape)
                    print(decodedwgt)

                    np.save('{}/{}'.format(output_path, layer.attrib['name'].replace('/', '_')), decodedwgt)


def main():
    print('*** OpenVINO IR model weight data extractor')
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='input IR model path')
    parser.add_argument('-o', '--output_path', type=str, help='weights output folder path')
    args = parser.parse_args()

    model, ext = os.path.splitext(args.model)
    output_path = args.output_path.rstrip('/')
    if ext != '.xml':
        print('The specified model is not \'.xml\' file')
        sys.exit(-1)
    dumpWeight(model, output_path)

if __name__ == "__main__":
    main()
