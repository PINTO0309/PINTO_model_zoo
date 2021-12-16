#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1yCdRJL0LrR3cYdKXfvX764SUZ6-vA1aF" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1yCdRJL0LrR3cYdKXfvX764SUZ6-vA1aF" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
