#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1O64WzXVnnt_CXelvqdibCDGnN1Eq1BBK" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1O64WzXVnnt_CXelvqdibCDGnN1Eq1BBK" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
