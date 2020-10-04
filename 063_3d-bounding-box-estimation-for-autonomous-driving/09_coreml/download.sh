#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1l-AMKUqn5kleNNZ4TektNes3hJV8aHn8" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1l-AMKUqn5kleNNZ4TektNes3hJV8aHn8" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
