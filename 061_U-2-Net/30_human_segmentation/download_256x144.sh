#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1qu0iZ0K11Kwu8bdC9ls47A-y4ROWqkFe" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1qu0iZ0K11Kwu8bdC9ls47A-y4ROWqkFe" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
