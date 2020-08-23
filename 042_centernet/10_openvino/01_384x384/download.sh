#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Uy-AuTr7uvurB8atqUtjadDJJn_6Eaub" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Uy-AuTr7uvurB8atqUtjadDJJn_6Eaub" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
