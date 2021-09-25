#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1gb-aDZ5bmBIxtuQSpr4FNl9qYKWB5itp" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1gb-aDZ5bmBIxtuQSpr4FNl9qYKWB5itp" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
