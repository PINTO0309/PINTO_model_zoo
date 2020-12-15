#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Y6B6l72hib9t5ammT-7ya-muBzRKZ7xq" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Y6B6l72hib9t5ammT-7ya-muBzRKZ7xq" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
