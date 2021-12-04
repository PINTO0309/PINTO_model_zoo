#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1V3ivWAkr86OHVc4e5WbJm_Elm_mt5DHe" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1V3ivWAkr86OHVc4e5WbJm_Elm_mt5DHe" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
