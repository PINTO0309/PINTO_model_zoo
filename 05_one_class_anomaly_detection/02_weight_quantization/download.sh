#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1JSFSfgwI1zP9JanSTbyY4NgRkjPVlSvU" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1JSFSfgwI1zP9JanSTbyY4NgRkjPVlSvU" -o weights_weight_quant.tflite

echo Download finished.
