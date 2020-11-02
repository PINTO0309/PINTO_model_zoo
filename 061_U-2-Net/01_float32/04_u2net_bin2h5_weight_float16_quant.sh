#!/bin/bash

pip3 install openvino2tensorflow --upgrade

openvino2tensorflow \
--model_path openvino/256x256/FP32/u2netp_256x256.xml \
--model_output_path saved_model_256x256 \
--output_saved_model True \
--output_h5 True \
--output_pb True \
--output_no_quant_float32_tflite True \
--output_weight_quant_tflite True \
--output_float16_quant_tflite True


openvino2tensorflow \
--model_path openvino/320x320/FP32/u2netp_320x320.xml \
--model_output_path saved_model_320x320 \
--output_saved_model True \
--output_h5 True \
--output_pb True \
--output_no_quant_float32_tflite True \
--output_weight_quant_tflite True \
--output_float16_quant_tflite True


openvino2tensorflow \
--model_path openvino/480x640/FP32/u2netp_480x640.xml \
--model_output_path saved_model_480x640 \
--output_saved_model True \
--output_h5 True \
--output_pb True \
--output_no_quant_float32_tflite True \
--output_weight_quant_tflite True \
--output_float16_quant_tflite True
