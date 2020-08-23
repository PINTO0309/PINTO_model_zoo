#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1JtwF4sPi4-a0PcT4mpxwQA5AYYMnO_rb" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1JtwF4sPi4-a0PcT4mpxwQA5AYYMnO_rb" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
