#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=19occYMVE7KkB7LGoLrAJDdBzd78oRKu8" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=19occYMVE7KkB7LGoLrAJDdBzd78oRKu8" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
