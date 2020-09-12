#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=18hoDWq0AAqMnbOZ1kNP81W0-P8YrlaqW" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=18hoDWq0AAqMnbOZ1kNP81W0-P8YrlaqW" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
