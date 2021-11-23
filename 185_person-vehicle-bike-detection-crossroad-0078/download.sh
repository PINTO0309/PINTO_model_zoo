#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=12mUKory2Q_Mp1qYVB0LRMaCRwzTBKKO-" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=12mUKory2Q_Mp1qYVB0LRMaCRwzTBKKO-" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
