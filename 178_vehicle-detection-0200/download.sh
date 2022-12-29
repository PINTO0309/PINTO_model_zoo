#!/bin/bash

curl "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/178_vehicle-detection-0200/resources.tar.gz" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
