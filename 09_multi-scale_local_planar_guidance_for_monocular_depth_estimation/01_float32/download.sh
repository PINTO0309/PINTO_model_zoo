#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1hmzINS-NNDxBQlpMGvAbj837t4d7ob3X" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1hmzINS-NNDxBQlpMGvAbj837t4d7ob3X" -o bts_densenet161_480_640.tar.gz
tar -zxvf bts_densenet161_480_640.tar.gz
rm bts_densenet161_480_640.tar.gz

echo Download finished.
