#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=u7Ez4HMLI5A5ite9eIH7JtG0ykJPeBo" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=u7Ez4HMLI5A5ite9eIH7JtG0ykJPeBo" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
