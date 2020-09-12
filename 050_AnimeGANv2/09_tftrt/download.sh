#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=19kE2qufA6qqX3rtBATNZsG6nCl5-6cAE" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=19kE2qufA6qqX3rtBATNZsG6nCl5-6cAE" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
