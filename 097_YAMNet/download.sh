#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=15vIIjX5gFmBzFoB0Xqj6vjtn06L29C0_" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=15vIIjX5gFmBzFoB0Xqj6vjtn06L29C0_" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
