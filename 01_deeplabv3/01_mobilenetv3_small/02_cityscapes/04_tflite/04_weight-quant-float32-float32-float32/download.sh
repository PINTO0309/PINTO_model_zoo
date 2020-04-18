#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1JzYLv7M0KVdLsqwu5lvRFEpUQv68hryk" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1JzYLv7M0KVdLsqwu5lvRFEpUQv68hryk" -o deeplab_mnv3_small_weight_quant_257.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1HaZMOrJixipNz7Ia23xZh8C65v007mNe" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1HaZMOrJixipNz7Ia23xZh8C65v007mNe" -o deeplab_mnv3_small_weight_quant_769.tflite

echo Download finished.
