#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=13TfdW0SKBL6c8w7LEq5zmvUyYySUHqHh" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=13TfdW0SKBL6c8w7LEq5zmvUyYySUHqHh" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
