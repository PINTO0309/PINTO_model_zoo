#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1rxyI7nJQeY_qez5l_x0RyL9yWLVuAzw_" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1rxyI7nJQeY_qez5l_x0RyL9yWLVuAzw_" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
