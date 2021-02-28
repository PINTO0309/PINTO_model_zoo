#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=11yNWHOg0TdRjT5_fYjTJa8lkJ7ILgYX4" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=11yNWHOg0TdRjT5_fYjTJa8lkJ7ILgYX4" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1LlO-L1Hmta5uPrQVZainJslgoGBfw0ex" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1LlO-L1Hmta5uPrQVZainJslgoGBfw0ex" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
