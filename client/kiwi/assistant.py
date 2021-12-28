import argparse
import sys

import numpy as np

import torch
from kiwi.core import SyncTritonModel as TritonModel
from kiwi.audio import AudioInput, AudioOutput, list_audio_devices
from kiwi.audio import audio_to_float, audio_to_int16

from kiwi.utils import download_or_load
from kiwi.vad import VoiceActivityDetection

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

import librosa

import time
import scipy


CHUNK = 8192
INPUT_SAMPLERATE = 16000
OUTPUT_SAMPLERATE = 22050

parser = argparse.ArgumentParser()
parser.add_argument(
    "--wav",
    default=None,
    type=str,
    help="path to input wav/ogg/flac file",
)
parser.add_argument(
    "--mic",
    default=None,
    type=str,
    help="device name or number of input microphone",
)
parser.add_argument(
    "--output-device",
    default=None,
    type=str,
    help="device name or number of audio output",
)
parser.add_argument(
    "--list-devices",
    action="store_true",
    help="list audio input devices",
)

parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    required=False,
    default=False,
    help="Enable verbose output",
)
parser.add_argument(
    "-u",
    "--url",
    type=str,
    required=False,
    default="localhost:8001",
    help="Inference server URL. Default is localhost:8001.",
)
parser.add_argument(
    "-s",
    "--ssl",
    action="store_true",
    required=False,
    default=False,
    help="Enable SSL encrypted channel to the server",
)
parser.add_argument(
    "-t",
    "--client-timeout",
    type=float,
    required=False,
    default=None,
    help="Client timeout in seconds. Default is None.",
)
parser.add_argument(
    "-r",
    "--root-certificates",
    type=str,
    required=False,
    default=None,
    help="File holding PEM-encoded root certificates. Default is None.",
)
parser.add_argument(
    "-p",
    "--private-key",
    type=str,
    required=False,
    default=None,
    help="File holding PEM-encoded private key. Default is None.",
)
parser.add_argument(
    "-x",
    "--certificate-chain",
    type=str,
    required=False,
    default=None,
    help="File holding PEM-encoded certicate chain. Default is None.",
)
parser.add_argument(
    "-C",
    "--grpc-compression-algorithm",
    type=str,
    required=False,
    default=None,
    help="The compression algorithm to be used when sending request to server. Default is None.",
)


def main(args):
    if args.list_devices:
        list_audio_devices()
        sys.exit()

    audio_input = AudioInput(
        wav=args.wav,
        mic=args.mic,
        sample_rate=INPUT_SAMPLERATE,
        chunk_size=CHUNK,
    )

    audio_output = AudioOutput(
        device=args.output_device,
        sample_rate=OUTPUT_SAMPLERATE,
    )

    try:
        client = grpcclient.InferenceServerClient(
            url=args.url,
            verbose=args.verbose,
            ssl=args.ssl,
            root_certificates=args.root_certificates,
            private_key=args.private_key,
            certificate_chain=args.certificate_chain,
        )
    except InferenceServerException as e:
        print(f"failed to creat context {str(e)}")
        sys.exit(1)

    assistant = TritonModel("assistant", client)
    asr = TritonModel("asr", client)
    # nlp = TritonModel("nlp", client)
    # tts = TritonModel("tts", client)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    vad_model_path = download_or_load(
        "misc/vad.pt",
        lang="multi",
    )
    vad = VoiceActivityDetection(
        model_path=vad_model_path,
        device=device,
    )
    print(f"vad running on {device}")

    speech_section = []
    transcript = ""
    for chunk in audio_input:
        print("listening...")
        chunk = np.array(chunk).astype(np.float32) / 32767

        speech_intervals = vad(chunk)
        # speech_intervals = _split_audio(chunk)

        if speech_intervals:
            speech_section.append(chunk)

        if len(speech_section) > 2:
            speech = np.hstack(speech_section)
            transcript = asr(speech)

            print(len(speech_section))
            print(f"transcript: {transcript}")

        if len(speech_section) > 2 and transcript != "":
            print("answering...")

            audio_input.close()

            speech = np.hstack(speech_section)
            speech_section = []

            wav = assistant(speech)
            wav = wav / 32767

            audio_output.write(wav)

            """
            wav = speech
            wav = librosa.resample(wav, INPUT_SAMPLERATE, OUTPUT_SAMPLERATE)
            audio_output.write(wav)
            """

            """
            answer = nlp(np.array([transcript.encode("utf-8")]).astype(object))
            print(f"question: {transcript}, answer: {answer}")
            wav = tts(np.array([answer.encode("utf-8")]).astype(object))

            # wav = librosa.resample(wav, OUTPUT_SAMPLERATE, INPUT_SAMPLERATE)

            audio_output.write(wav / 32767)
            """

            time.sleep(0.5)
            audio_input.open()


def _split_audio(signal: np.ndarray, top_db: int = 48) -> list:
    speech_intervals = list()
    start, end = 0, 0

    non_silence_indices = librosa.effects.split(signal, top_db=top_db)

    for _, end in non_silence_indices:
        speech_intervals.append(signal[start:end])
        start = end

    speech_intervals.append(signal[end:])

    return speech_intervals


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
