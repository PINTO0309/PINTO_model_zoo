#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=196BSzeFT9t6VdFqD6UYgsxc7lGXPbm_p" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=196BSzeFT9t6VdFqD6UYgsxc7lGXPbm_p" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
