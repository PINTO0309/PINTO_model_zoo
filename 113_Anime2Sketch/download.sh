#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=19O16mN8iWv0BY8QPsuXpeH74lFFD5zWL" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=19O16mN8iWv0BY8QPsuXpeH74lFFD5zWL" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
