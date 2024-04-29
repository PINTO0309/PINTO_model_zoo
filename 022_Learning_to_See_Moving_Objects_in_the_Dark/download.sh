#!/bin/bash

curl "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/022_Learning_to_See_Moving_Objects_in_the_Dark/022_Learning_to_See_Moving_Objects_in_the_Dark.tar.gz" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

echo Download finished.
