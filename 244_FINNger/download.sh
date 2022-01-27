#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=18_jqEgz4Guh8rZwXGn8cBe-VmEx7aysi" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=18_jqEgz4Guh8rZwXGn8cBe-VmEx7aysi" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
