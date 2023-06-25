#!/bin/bash

curl "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/110_L-CNN/resources.tar.gz" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
