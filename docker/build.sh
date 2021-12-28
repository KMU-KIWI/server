#!/bin/bash
set -e

docker build -t tritonclient -f docker/Dockerfile.client ./client
docker build -t tritonserver -f docker/Dockerfile.server ./docker

docker build -t tritondev:server -f docker/Dockerfile.dev-server ./docker
docker build -t tritondev:client -f docker/Dockerfile.dev-client ./docker
