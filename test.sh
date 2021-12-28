#!/bin/bash

set -e

source docker/build.sh

docker run --rm \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    --name tritonserver \
    -v $(pwd)/models:/models \
    -v $(pwd)/data:/data \
    -d --init tritonserver tritonserver --model-repository=/models \
                                        --log-verbose 1 \
                                        --log-info true \
                                        --log-warning true \
                                        --log-error true \
                                        --exit-on-error false

timeout=300
t=0
while [ $t -lt "$timeout" ]; do
    if [ $(curl -sL -w "%{http_code}\\n" localhost:8000/v2/health/ready -o /dev/null) == "200" ];
    then
        set +e
        docker run --rm --net host -v $(pwd)/tests:/tests tritonclient \
            python3 -m unittest discover -s /tests

        echo "stopping server"
        docker stop tritonserver
        exit
    else
        t=$((t+1))
        sleep 1
    fi
done

echo "tritonserver is not ready"

docker logs tritonserver

echo "stopping server"
docker stop tritonserver

trap "docker stop tritonserver" SIGINT
