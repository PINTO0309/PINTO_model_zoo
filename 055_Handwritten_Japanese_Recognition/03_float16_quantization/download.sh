#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1s2DPG-4iDfhqjiYa5lVhZsRc4nUlmQ-z" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1s2DPG-4iDfhqjiYa5lVhZsRc4nUlmQ-z" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
