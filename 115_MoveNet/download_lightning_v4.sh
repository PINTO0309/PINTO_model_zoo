#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Ss84uHlr3RjdIq3W36Ir2QE0DGzJIkuJ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Ss84uHlr3RjdIq3W36Ir2QE0DGzJIkuJ" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
