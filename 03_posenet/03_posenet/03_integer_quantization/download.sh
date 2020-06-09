#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1W20EJzKlICMV2F1q13oFN07rYPUTJc2V" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1W20EJzKlICMV2F1q13oFN07rYPUTJc2V" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
