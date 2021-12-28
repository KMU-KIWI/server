import time
import sys
from typing import Dict

import numpy as np

import tritonclient.grpc as grpcclient

from tritonclient import utils

# import tritonclient.utils.shared_memory as shm
from tritonclient.utils import triton_to_np_dtype, np_to_triton_dtype
from tritonclient.utils import InferenceServerException


class ParseTritonModel:
    def __init__(self, model_name: str, client):
        """
        model_metadata:

        input [
          {
            name: "input0"
            data_type: TYPE_FP32
            dims: [ 16 ]
          },
          {
            name: "input1"
            data_type: TYPE_FP32
            dims: [ 16 ]
          }
        ]
        output [
          {
            name: "output0"
            data_type: TYPE_FP32
            dims: [ 16 ]
          }
        ]
        """

        self.model_name = model_name
        self.client = client

        try:
            model_metadata = client.get_model_metadata(model_name, as_json=True)
        except InferenceServerException as e:
            print(f"failed to get {model_name} model metadata: {str(e)}")
            sys.exit(1)

        self.input_metadata = model_metadata["inputs"]
        self.output_metadata = model_metadata["outputs"]

    def generate_input(self, *inputs):
        return self._generate_input(*inputs)

    def _generate_input(self, *inputs):
        infer_inputs = []
        for metadata, np_input in zip(self.input_metadata, inputs):
            infer_input = grpcclient.InferInput(
                metadata["name"],
                np_input.shape,
                metadata["datatype"],
            )
            infer_input.set_data_from_numpy(np_input)

            infer_inputs.append(infer_input)
        return infer_inputs

    def generate_req_output(self):
        return self._generate_req_output()

    def _generate_req_output(self):
        req_output = []
        for metadata in self.output_metadata:
            req_output.append(grpcclient.InferRequestedOutput(metadata["name"]))
        return req_output

    def parse_infer_result(self, infer_result):
        return self._parse_infer_result(infer_result)

    def _parse_infer_result(self, infer_result):
        result = []
        for metadata in self.output_metadata:
            parsed_result = infer_result.as_numpy(metadata["name"])
            if metadata["datatype"] == "BYTES":
                parsed_result = list(
                    map(lambda t: t.decode("utf-8"), parsed_result.tolist())
                )
                if len(parsed_result) == 1:
                    parsed_result = parsed_result[0]
            result.append(parsed_result)
        if len(result) == 1:
            return result[0]
        else:
            return result


class SyncTritonModel(ParseTritonModel):
    def __init__(self, model_name, client):
        super().__init__(model_name, client)

    def __call__(self, *inputs):
        infer_result = self.client.infer(
            model_name=self.model_name,
            inputs=self.generate_input(*inputs),
            outputs=self.generate_req_output(),
        )
        return self.parse_infer_result(infer_result)


class AsyncTritonModel(ParseTritonModel):
    def __init__(self, model_name, client):
        super().__init__(model_name, client)

    def __call__(self, *inputs):
        inputs = self.generate_input(*inputs)
        outputs = self.generate_req_output()
        self.client.async_infer()


"""
class ShmSyncTritonModel(SyncTritonModel):
    def __init__(self, model_name, client):
        super().__init__(model_name=model_name, client=client)

        self.output_in_shm = False

    def generate_input(self, *inputs):
        handles = []
        byte_sizes = []
        for input0, metadata in zip(inputs, self.input_metadata):
            if metadata["datatype"] == "BYTES":
                input0_data_serialized = utils.serialize_byte_tensor(input0)
                input0_byte_size = utils.serialize_byte_size(input0_data_serialized)
            else:
                input0_byte_size = input0.size * input0.data.itemsize

            name = metadata["name"]
            key = f"/{name}"
            shm.create_shared_memory_region(name, key, input0_byte_size)
            handles.append(
                self.client.register_system_shared_memory(name, key, input0_byte_size)
            )
            byte_sizes.append(input0_byte_size)

        self.shm_in_handles = handles

        inputs = self._generate_input()
        for i, (byte_size, metadata) in enumerate(zip(byte_sizes, self.input_metadata)):
            inputs[i].set_shared_memory(metadata["name"], byte_size)

        return inputs
"""
