#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1V2QSu998wDSP39H5x17t0_h3J2chQ1ns" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1V2QSu998wDSP39H5x17t0_h3J2chQ1ns" -o resources.tar.gz

tar -zxvf resources.tar.gz
rm resources.tar.gz
echo Download finished.
