import unittest
import sys
import time

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype, np_to_triton_dtype

from contextlib import contextmanager
from tempfile import NamedTemporaryFile

from requests import get
from pydub import AudioSegment

import numpy as np

from kiwi.core import SyncTritonModel as TritonModel


# from pororo/utils.py
@contextmanager
def control_temp(file_path: str):
    """
    Download temporary file from web, then remove it after some context
    Args:
        file_path (str): web file path
    """
    # yapf: disable
    assert file_path.startswith("http"), "File path should contain `http` prefix !"
    # yapf: enable

    ext = file_path[file_path.rfind(".") :]

    with NamedTemporaryFile("wb", suffix=ext, delete=True) as f:
        response = get(file_path, allow_redirects=True)
        f.write(response.content)
        yield f.name


class TritonTester(unittest.TestCase):
    def setUp(self):
        self.client = grpcclient.InferenceServerClient(
            url="localhost:8001",
            verbose=False,
        )

    def test_asr(self):
        asr = TritonModel("asr", self.client)

        with control_temp(
            "https://twg.kakaocdn.net/pororo/ko/example/korean_speech.wav"
        ) as f_src:
            wav = AudioSegment.from_wav(f_src)

        channel_sounds = wav.split_to_mono()
        wav = np.array([s.get_array_of_samples() for s in channel_sounds])[0] / 32767.0

        text = asr(wav.astype(np.float32))
        self.assertIsInstance(text, str)

    def test_nlp(self):
        nlp = TritonModel("nlp", self.client)

        answer = nlp(np.array(["안녕하세요"]).astype(object))[0]
        self.assertIsInstance(answer, str)

    def test_tts(self):
        tts = TritonModel("tts", self.client)

        wav = tts(np.array(["안녕하세요"]).astype(object))
        self.assertIsInstance(wav, np.ndarray)


if __name__ == "__main__":
    unittest.main()
