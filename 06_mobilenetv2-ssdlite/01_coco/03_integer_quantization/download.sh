#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1LjTqn5nChAVKhXgwBUp00XIKXoZrs9sB" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1LjTqn5nChAVKhXgwBUp00XIKXoZrs9sB" -o ssdlite_mobilenet_v2_coco_300_integer_quant.tflite

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1poH1bnh_4UYbYoNtunDMJPvLMMZng4VQ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1poH1bnh_4UYbYoNtunDMJPvLMMZng4VQ" -o ssdlite_mobilenet_v2_coco_300_integer_quant_with_postprocess.tflite

echo Download finished.
