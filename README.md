# Kiwi Server

Kiwi Server is an infernce server for conversational AI with ASR, NLP, TTS componenets. Kiwi Server uses the [Triton Inference Server](https://github.com/triton-inference-server/server) for serving client requests on the cloud. Kiwi Server uses [Pororo](https://github.com/kakaobrain/pororo/tree/7d05a75e8062b00e6b65364b8ec6c52b6293ab07), for natural language processing and speech related tasks.

# Running the Container

## build the images

running `./docker/build.sh` will build the server and client images

## start the server

```
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
```

## start the client

```
docker run --rm --net host -v $(pwd)/tests:/tests tritonclient \
    python3 -m unittest discover -s /tests
```

# testing

run `./test.sh`
