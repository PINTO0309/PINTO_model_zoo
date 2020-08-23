#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1yjphe3qLFxH0R-U6_3YaGZBggfKTejqZ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1yjphe3qLFxH0R-U6_3YaGZBggfKTejqZ" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
