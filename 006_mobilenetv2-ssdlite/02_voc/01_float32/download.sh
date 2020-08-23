#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=12FKxTMcz65I-RmY9yDvj0IIr2oP8Hars" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=12FKxTMcz65I-RmY9yDvj0IIr2oP8Hars" -o tflite_graph_with_postprocess.pb

echo Download finished.
