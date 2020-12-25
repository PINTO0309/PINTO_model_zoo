#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1V-AD7iO5ZpxXYVwZ2DD2AsrqhY8W2ZF3" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1V-AD7iO5ZpxXYVwZ2DD2AsrqhY8W2ZF3" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
