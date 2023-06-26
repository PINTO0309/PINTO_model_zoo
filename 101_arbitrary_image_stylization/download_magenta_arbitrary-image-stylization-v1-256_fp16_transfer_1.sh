#!/bin/bash

curl "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/101_arbitrary_image_stylization/magenta_arbitrary-image-stylization-v1-256_fp16_transfer_1/resources.tar.gz" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
