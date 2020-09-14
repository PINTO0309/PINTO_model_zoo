#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1V4ajBQhk7Rk5ku_1_2lVJ4xthXknW7_Y" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1V4ajBQhk7Rk5ku_1_2lVJ4xthXknW7_Y" -o resources.tar.gz

tar -zxvf resources.tar.gz
rm resources.tar.gz
echo Download finished.
