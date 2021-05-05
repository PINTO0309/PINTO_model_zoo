#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1UrIZXXJHZJ_X5XWOhz1E3n2e1hnfIwra" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1UrIZXXJHZJ_X5XWOhz1E3n2e1hnfIwra" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
