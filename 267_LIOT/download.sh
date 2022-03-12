#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=18933w0L4AwT16fiLXoWXMUK98yyTqyEV" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=18933w0L4AwT16fiLXoWXMUK98yyTqyEV" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
