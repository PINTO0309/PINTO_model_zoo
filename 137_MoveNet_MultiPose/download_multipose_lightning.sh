#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1DR8i-8uFg3HaVkhziTDRR03QJC5aaf4d" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1DR8i-8uFg3HaVkhziTDRR03QJC5aaf4d" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
