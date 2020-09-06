#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1F3INx_31E-z3wN3jshdc_Ck8bSm4t2AB" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1F3INx_31E-z3wN3jshdc_Ck8bSm4t2AB" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
