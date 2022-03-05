#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1B1730O3yTzFiqUa31adDgG6QxbF0BS8Q" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1B1730O3yTzFiqUa31adDgG6QxbF0BS8Q" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
