#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=17P_emMbxdT0anYSJRC8kZcZDJA8kP_Tw" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=17P_emMbxdT0anYSJRC8kZcZDJA8kP_Tw" -o resources.tar.gz

tar -zxvf resources.tar.gz
rm resources.tar.gz
echo Download finished.
