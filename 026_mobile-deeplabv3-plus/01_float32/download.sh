#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=14I2irGlVQhoS5bFC1sOi6sIbUTumVdrU" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=14I2irGlVQhoS5bFC1sOi6sIbUTumVdrU" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
