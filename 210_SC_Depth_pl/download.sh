#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1QPWuL21UI3-kNuYr6OAQfGvzaCBbc01z" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1QPWuL21UI3-kNuYr6OAQfGvzaCBbc01z" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
