#!/bin/bash

RELEASE=v1.0.0

curl -L https://github.com/fabio-sim/DocShadow-ONNX-TensorRT/releases/download/${RELEASE}/docshadow_sd7k.onnx -o weights/docshadow_sd7k.onnx
curl -L https://github.com/fabio-sim/DocShadow-ONNX-TensorRT/releases/download/${RELEASE}/docshadow_jung.onnx -o weights/docshadow_jung.onnx
curl -L https://github.com/fabio-sim/DocShadow-ONNX-TensorRT/releases/download/${RELEASE}/docshadow_kligler.onnx -o weights/docshadow_kligler.onnx
