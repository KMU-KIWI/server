import numpy as np
import sys
import json

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

import logging
import os
from typing import Optional

import torch

from pororo.tasks.utils.base import PororoFactoryBase, PororoSimpleBase
from pororo.tasks.utils.download_utils import download_or_load


class KiwiFactory(PororoFactoryBase):
    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["ko"]

    @staticmethod
    def get_available_models():
        return {
            "ko": ["wav2vec.ko"],
        }

    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if self.config.lang not in self.get_available_langs():
            raise ValueError(
                f"Unsupported Language : {self.config.lang}",
                'Support Languages : ["ko"]',
            )
        from pororo.models.wav2vec2.recognizer import BrainWav2Vec2Recognizer

        model_path = download_or_load(
            f"misc/{self.config.n_model}.pt",
            self.config.lang,
        )
        dict_path = download_or_load(
            f"misc/{self.config.lang}.ltr.txt",
            self.config.lang,
        )
        vad_model_path = download_or_load(
            "misc/vad.pt",
            lang="multi",
        )

        try:
            import librosa  # noqa

            logging.getLogger("librosa").setLevel(logging.WARN)
        except ModuleNotFoundError as error:
            raise error.__class__("Please install librosa with: `pip install librosa`")

        from pororo.models.vad import VoiceActivityDetection

        vad_model = VoiceActivityDetection(
            model_path=vad_model_path,
            device=device,
        )

        model = BrainWav2Vec2Recognizer(
            model_path=model_path,
            dict_path=dict_path,
            vad_model=vad_model,
            device=device,
            lang=self.config.lang,
        )
        return Kiwi(model, self.config)


class Kiwi(PororoSimpleBase):
    def __init__(self, model, config):
        super().__init__(config)
        self._model = model
        self.SAMPLE_RATE = 16000
        self.MAX_VALUE = 32767

    def predict(
        self,
        signal: np.ndarray,
        **kwargs,
    ) -> dict:
        """
        Conduct speech recognition for audio in a given path

        Args:
            audio_path (str): the wav file path
            top_db (int): the threshold (in decibels) below reference to consider as silence (default: 48)
            batch_size (int): inference batch size (default: 1)
            vad (bool): flag indication whether to use voice activity detection or not, If it is False, it is split into
             dB criteria and then speech recognition is made. Applies only when audio length is more than 50 seconds.

        Returns:
            dict: result of speech recognition

        """
        top_db = kwargs.get("top_db", 48)
        batch_size = kwargs.get("batch_size", 1)
        vad = kwargs.get("batch_size", False)

        return self._model.predict(
            audio_path="",
            signal=signal,
            top_db=top_db,
            vad=vad,
            batch_size=batch_size,
        )


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "RECOGNIZED_TEXT"
        )

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

        # 내 클래스 생성해서 load 하기's
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = KiwiFactory("asr", "ko", None).load(device)
        print("Initialized...")

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
        for request in requests:
            # Perform inference on the request and append it to responses list...
            input0 = pb_utils.get_input_tensor_by_name(request, "RAW_SPEECH")

            results = self.model(input0.as_numpy())["results"]
            output0 = np.array([result["speech"].encode("utf-8") for result in results])

            if output0.size == 0:
                output0 = np.array(["".encode("utf-8")])

            output0 = pb_utils.Tensor(
                "RECOGNIZED_TEXT", output0.astype(self.output0_dtype)
            )

            inference_response = pb_utils.InferenceResponse(output_tensors=[output0])
            responses.append(inference_response)

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
