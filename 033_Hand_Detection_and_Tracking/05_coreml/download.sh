#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=18Z4b4Wyo1stgaxGG7K32tZPt47RhvUSI" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=18Z4b4Wyo1stgaxGG7K32tZPt47RhvUSI" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
