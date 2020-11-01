#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1o3yr2rxdAnO3rIo0F0DZxbJk-dbPP72H" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1o3yr2rxdAnO3rIo0F0DZxbJk-dbPP72H" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
