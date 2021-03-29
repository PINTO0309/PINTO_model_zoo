#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1HdJ1UaK48qCSGEPC-xLFXCOrlS8QM5a4" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1HdJ1UaK48qCSGEPC-xLFXCOrlS8QM5a4" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
