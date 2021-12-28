"""Module download related function from. Tenth"""

import logging
import os
import platform
import sys
import zipfile
from dataclasses import dataclass
from typing import Tuple, Union

import wget

DEFAULT_PREFIX = {
    "model": "https://twg.kakaocdn.net/pororo/{lang}/models",
    "dict": "https://twg.kakaocdn.net/pororo/{lang}/dicts",
}


@dataclass
class DownloadInfo:
    r"Download information such as defined directory, language and model name"
    n_model: str
    lang: str
    root_dir: str


def get_save_dir(save_dir: str = None) -> str:
    """
    Get default save directory
    Args:
        savd_dir(str): User-defined save directory
    Returns:
        str: Set save directory
    """
    # If user wants to manually define save directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    pf = platform.system()

    if pf == "Windows":
        save_dir = "C:\\pororo"
    else:
        home_dir = os.path.expanduser("~")
        save_dir = os.path.join(home_dir, ".pororo")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir


def get_download_url(n_model: str, key: str, lang: str) -> str:
    """
    Get download url using default prefix
    Args:
        n_model (str): model name
        key (str): key name either `model` or `dict`
        lang (str): language name
    Returns:
        str: generated download url
    """
    default_prefix = DEFAULT_PREFIX[key].format(lang=lang)
    return f"{default_prefix}/{n_model}"


def download_or_load_bert(info: DownloadInfo) -> str:
    """
    Download fine-tuned BrainBert & BrainSBert model and dict
    Args:
        info (DownloadInfo): download information
    Returns:
        str: downloaded bert & sbert path
    """
    model_path = os.path.join(info.root_dir, info.n_model)

    if not os.path.exists(model_path):
        info.n_model += ".zip"
        zip_path = os.path.join(info.root_dir, info.n_model)

        type_dir = download_from_url(
            info.n_model,
            zip_path,
            key="model",
            lang=info.lang,
        )

        zip_file = zipfile.ZipFile(zip_path)
        zip_file.extractall(type_dir)
        zip_file.close()

    return model_path


def download_or_load_misc(info: DownloadInfo) -> str:
    """
    Download (pre-trained) miscellaneous model
    Args:
        info (DownloadInfo): download information
    Returns:
        str: miscellaneous model path
    """
    # Add postfix <.model> for sentencepiece
    if "sentencepiece" in info.n_model:
        info.n_model += ".model"

    # Generate target model path using root directory
    model_path = os.path.join(info.root_dir, info.n_model)
    if not os.path.exists(model_path):
        type_dir = download_from_url(
            info.n_model,
            model_path,
            key="model",
            lang=info.lang,
        )

        if ".zip" in info.n_model:
            zip_file = zipfile.ZipFile(model_path)
            zip_file.extractall(type_dir)
            zip_file.close()

    if ".zip" in info.n_model:
        model_path = model_path[: model_path.rfind(".zip")]
    return model_path


def download_from_url(
    n_model: str,
    model_path: str,
    key: str,
    lang: str,
) -> str:
    """
    Download specified model from Tenth
    Args:
        n_model (str): model name
        model_path (str): pre-defined model path
        key (str): type key (either model or dict)
        lang (str): language name
    Returns:
        str: default type directory
    """
    # Get default type dir path
    type_dir = "/".join(model_path.split("/")[:-1])
    os.makedirs(type_dir, exist_ok=True)

    # Get download tenth url
    url = get_download_url(n_model, key=key, lang=lang)

    logging.info("Downloading user-selected model...")
    wget.download(url, type_dir)
    sys.stderr.write("\n")
    sys.stderr.flush()

    return type_dir


def download_or_load(
    n_model: str,
    lang: str,
    custom_save_dir: str = None,
) -> Union[str, Tuple[str, str]]:
    """
    Download or load model based on model information
    Args:
        n_model (str): model name
        lang (str): language information
        custom_save_dir (str, optional): user-defined save directory path. defaults to None.
    Returns:
        Union[TransformerInfo, str, Tuple[str, str]]
    """
    root_dir = get_save_dir(save_dir=custom_save_dir)
    info = DownloadInfo(n_model, lang, root_dir)

    return download_or_load_misc(info)
