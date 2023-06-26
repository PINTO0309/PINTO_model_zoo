#!/bin/bash

curl "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/093_ocr_japanese/saved_model_detectionnet.tar.gz" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

curl "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/093_ocr_japanese/saved_model_classifiernet.tar.gz" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
