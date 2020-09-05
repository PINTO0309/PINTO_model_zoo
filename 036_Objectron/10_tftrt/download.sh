#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1i0eCtzoWgNtBA8femcK7_1G_1ob9B2NX" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1i0eCtzoWgNtBA8femcK7_1G_1ob9B2NX" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
