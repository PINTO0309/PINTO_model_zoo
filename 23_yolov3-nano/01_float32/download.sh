#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=13z-6INu_203A97p_pMvsQux0qnu-WV8A" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=13z-6INu_203A97p_pMvsQux0qnu-WV8A" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
