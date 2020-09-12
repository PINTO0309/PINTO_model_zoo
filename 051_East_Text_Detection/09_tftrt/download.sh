#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=18mkVVL_cK5GCfv9b-jPb8QYhEKKBk1aG" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=18mkVVL_cK5GCfv9b-jPb8QYhEKKBk1aG" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
