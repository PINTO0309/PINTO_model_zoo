#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1197PaaAUGHAJ_L_fQP5rzPqpCrk62qFM" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1197PaaAUGHAJ_L_fQP5rzPqpCrk62qFM" -o deeplab_mnv3_small_cityscapes_trainfine_769.pb

echo Download finished.
