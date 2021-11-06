#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1ywRiaO86keczCWdPj993zP3lpWd7hpAJ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1ywRiaO86keczCWdPj993zP3lpWd7hpAJ" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
