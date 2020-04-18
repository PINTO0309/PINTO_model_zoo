#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1GE5Jg8WfgxSzvcPpla5Crwoliocgkbq_" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1GE5Jg8WfgxSzvcPpla5Crwoliocgkbq_" -o v3-large-minimalistic_224_1.0_float.pb

echo Download finished.
