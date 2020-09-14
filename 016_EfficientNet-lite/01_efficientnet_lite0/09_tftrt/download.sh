#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1lYjcAcCV0jQunDugKQPBG0-6jS098LpH" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1lYjcAcCV0jQunDugKQPBG0-6jS098LpH" -o resources.tar.gz

tar -zxvf resources.tar.gz
rm resources.tar.gz
echo Download finished.
