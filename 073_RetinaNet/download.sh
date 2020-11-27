#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Wkg00lR9GJ4I_Yz2QwMRQdBHwwGNnqjd" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Wkg00lR9GJ4I_Yz2QwMRQdBHwwGNnqjd" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
