#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=12zOdg5BK4O775tKdhmQU7NZ9scJzKNUa" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=12zOdg5BK4O775tKdhmQU7NZ9scJzKNUa" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
