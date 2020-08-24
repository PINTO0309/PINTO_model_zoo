#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=10IbtcflBmik9SM1VixqQHeDaH0sEdbPW" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=10IbtcflBmik9SM1VixqQHeDaH0sEdbPW" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
