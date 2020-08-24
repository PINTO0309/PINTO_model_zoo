#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1KEeyVNZt2ZQak1j5386kEe0zKsC2rA3n" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1KEeyVNZt2ZQak1j5386kEe0zKsC2rA3n" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
