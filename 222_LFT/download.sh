#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=16pJIGf4Eq4qnGYj-5TnaaTwgV9EYRXJQ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=16pJIGf4Eq4qnGYj-5TnaaTwgV9EYRXJQ" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
