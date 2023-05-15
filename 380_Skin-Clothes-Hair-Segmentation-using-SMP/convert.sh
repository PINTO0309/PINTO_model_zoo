#!/bin/bash

onnx2tf -i schs_deeplabv3_512x512.onnx -cotof -osd
onnx2tf -i schs_pan_512x512.onnx -cotof -osd
onnx2tf -i schs_unet_plus_plus_512x512.onnx -cotof -osd