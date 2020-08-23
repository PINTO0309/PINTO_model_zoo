#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Cdmb_cb8IU2p6_xMWsW6SiQ1OgMSiOeJ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Cdmb_cb8IU2p6_xMWsW6SiQ1OgMSiOeJ" -o weights_integer_quant.tflite

echo Download finished.
