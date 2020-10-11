#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1PFbOhCW7Td6E0bQ2xbBIUmrog5jBbW-4" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1PFbOhCW7Td6E0bQ2xbBIUmrog5jBbW-4" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
