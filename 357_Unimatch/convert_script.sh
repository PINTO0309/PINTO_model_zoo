#!/bin/bash

onnx2tf -i gmflow-scale1-mixdata-train320x576-4c3a6e9a_1x3x480x640_bidir_flow_sim.onnx -osd -rtpo Erf -cotof -cotoa 1e-1
onnx2tf -i gmflow-scale1-mixdata-train320x576-4c3a6e9a_1x3x480x640_sim.onnx -osd -rtpo Erf -cotof -cotoa 1e-1
onnx2tf -i gmflow-scale1-things-e9887eda_1x3x480x640_bidir_flow_sim.onnx -osd -rtpo Erf -cotof -cotoa 1e-1
onnx2tf -i gmflow-scale1-things-e9887eda_1x3x480x640_sim.onnx -osd -rtpo Erf -cotof -cotoa 1e-1
onnx2tf -i gmstereo-scale1-resumeflowthings-sceneflow-16e38788_1x3x480x640_sim.onnx -osd -rtpo Erf -cotof -cotoa 1e-1
onnx2tf -i gmstereo-scale1-sceneflow-124a438f_1x3x480x640_sim.onnx -osd -rtpo Erf -cotof -cotoa 1e-1
