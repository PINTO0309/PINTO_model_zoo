#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Rx79CJISYhuDBHaX7A2yHtjJEbL8waog" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Rx79CJISYhuDBHaX7A2yHtjJEbL8waog" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
