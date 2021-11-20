#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1D5KwS6flOO5y7FJqXXLI7v_KFD4WUIgP" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1D5KwS6flOO5y7FJqXXLI7v_KFD4WUIgP" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
