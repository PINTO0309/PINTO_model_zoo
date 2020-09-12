#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=14ImepF-zyPvn7YaYZUH57-ETA361vRAu" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=14ImepF-zyPvn7YaYZUH57-ETA361vRAu" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
