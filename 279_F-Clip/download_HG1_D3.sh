#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1jMFrWrpXlv20g1zr6gRz6DvRlyY478Jp" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1jMFrWrpXlv20g1zr6gRz6DvRlyY478Jp" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
