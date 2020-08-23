#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=15sO6vgJGBjMhHZ76GY1IpMiFrLDQ4teB" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=15sO6vgJGBjMhHZ76GY1IpMiFrLDQ4teB" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
