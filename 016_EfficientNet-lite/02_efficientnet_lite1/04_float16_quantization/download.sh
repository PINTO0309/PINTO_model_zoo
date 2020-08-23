#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1LCNHQnQL9g7wKjD9iEo8sdjcNxB98p2d" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1LCNHQnQL9g7wKjD9iEo8sdjcNxB98p2d" -o resources.tar.gz

tar -zxvf resources.tar.gz
rm resources.tar.gz
echo Download finished.
