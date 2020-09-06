#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1g61Zpl93uThYS81TI3VkubCvWNZvgKYW" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1g61Zpl93uThYS81TI3VkubCvWNZvgKYW" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
