#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1gc-wWLUSm1SJh0-G-hb0bM0eUFmM4Ikd" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1gc-wWLUSm1SJh0-G-hb0bM0eUFmM4Ikd" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
