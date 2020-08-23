#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1byobKK7kfHuKumnada7rDSD1l2x9d6xO" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1byobKK7kfHuKumnada7rDSD1l2x9d6xO" -o saved_model.tar.gz
tar -zxvf saved_model.tar.gz
rm saved_model.tar.gz
echo Download finished.
