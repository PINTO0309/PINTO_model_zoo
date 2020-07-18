#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1_ZW8UHlO2Ana0ZdnmGFhUpXe29jSX3KD" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1_ZW8UHlO2Ana0ZdnmGFhUpXe29jSX3KD" -o resources.tar.gz

tar -zxvf resources.tar.gz
rm resources.tar.gz
echo Download finished.
