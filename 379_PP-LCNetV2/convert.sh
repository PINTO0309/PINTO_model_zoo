#!/bin/bash

onnx2tf -i PP-LCNetV2_1x3x224x224.onnx -cotof -coion -osd
onnx2tf -i PP-LCNetV2_Nx3x224x224.onnx -cotof -coion -osd