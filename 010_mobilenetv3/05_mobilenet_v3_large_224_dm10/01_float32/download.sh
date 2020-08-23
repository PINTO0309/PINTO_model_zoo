#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1oGFDm3TlvGna2PKaElOoUlADIAOYnYRI" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1oGFDm3TlvGna2PKaElOoUlADIAOYnYRI" -o v3-large_224_1.0_float.pb

echo Download finished.
