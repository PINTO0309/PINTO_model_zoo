#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1mlpr3y6fC3nPq92zJDvgpxtRZTQq2Bd_" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1mlpr3y6fC3nPq92zJDvgpxtRZTQq2Bd_" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
