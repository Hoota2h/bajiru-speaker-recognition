import pathlib

import pytest

from baji_recognition import process


@pytest.fixture
def data_dir():
    return pathlib.Path(__file__).parent / "data"


def test_low_voice(data_dir):
    low_voice = data_dir / "low.mp3"

    res = process.load_and_process(low_voice)

    assert res == "low"


def test_normal_voice(data_dir):
    normal_voice = data_dir / "normal.mp3"

    res = process.load_and_process(normal_voice)

    assert res == "high"
