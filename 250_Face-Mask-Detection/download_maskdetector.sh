#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Y_8S_2P9meYFg1Vw2_eWlVUFd5sXNMlv" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Y_8S_2P9meYFg1Vw2_eWlVUFd5sXNMlv" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
