#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1v3jj8VvQ9YZm0Se4XwnEmT4G41BljSPN" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1v3jj8VvQ9YZm0Se4XwnEmT4G41BljSPN" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
