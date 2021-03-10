#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1K7cSZCh0nJKgI2BKxwZNh2thU9oHNpTw" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1K7cSZCh0nJKgI2BKxwZNh2thU9oHNpTw" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
