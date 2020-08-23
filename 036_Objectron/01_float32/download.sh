#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=13W3K2jidsakvX5wudkvARdRcJ0dWV52K" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=13W3K2jidsakvX5wudkvARdRcJ0dWV52K" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
