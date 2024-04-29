#!/bin/bash

curl "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/012_Fast_Accurate_and_Lightweight_Super-Resolution/012_Fast_Accurate_and_Lightweight_Super-Resolution.tar.gz" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
