import pathlib

import librosa
import numpy as np
from matplotlib import pyplot as plt
from moviepy import AudioFileClip, CompositeVideoClip, ImageClip
from PIL import Image, ImageColor

from baji_recognition import identifier

enum = {"unknown": -1, "low": 0, "high": 1}
logger = identifier.logger
BAJI = "#19582b"
LOWJI = "#8fdd77"


def load_and_process(file: pathlib.Path) -> str:
    in_data = librosa.load(file, sr=None)[0]
    data = np.frombuffer(in_data, dtype=np.float32)
    return identifier.process(data, identifier.RATE)[0]


def chunk_file(file: pathlib.Path, chunk_length: float) -> tuple[list[np.ndarray], list[float]]:
    sr = librosa.get_samplerate(file)
    in_data, rate = librosa.load(file, sr=sr)
    data = np.frombuffer(in_data, dtype=np.float32)
    samples_per_chunk = int(chunk_length * rate)
    chunked_data = [data[i : i + samples_per_chunk] for i in range(0, len(data), samples_per_chunk)]

    times = [chunk_length] * len(chunked_data)
    times[-1] = chunked_data[-1].shape[0] / samples_per_chunk
    return chunked_data, times


def get_decisions(file: pathlib.Path, chunk_length: float) -> tuple[list[str], list[float]]:
    chunks, times = chunk_file(file, chunk_length)
    decisions = [identifier.process(chunk, identifier.RATE)[0] for chunk in chunks]
    return decisions, times


def get_values(file: pathlib.Path, chunk_length: float) -> list[tuple[float, float]]:
    return [identifier.process(chunk, identifier.RATE)[1:] for chunk in chunk_file(file, chunk_length)[0]]


def make_decision(decisions: list[str]) -> list[str]:
    # default to high
    curr_dec = "high"

    new_decisions = []
    for dec in decisions:
        decision = dec if dec != "unknown" else curr_dec
        new_decisions.append(decision)
    return new_decisions


def decision_to_num(decisions: list[str]) -> list[int]:
    return [enum[x] for x in decisions]


def plot_series(
    first_series: np.ndarray, first_label: str, second_series: np.ndarray, second_label: str, title: str
) -> None:
    plt.figure()
    plt.plot(first_series, label=first_label)
    plt.plot(second_series, label=second_label)
    plt.legend()
    plt.title(title)


def plot_decisions(file: pathlib.Path, label: str, title: str, chunk_length: float) -> None:
    plt.figure()
    preliminary_decisions, times = get_decisions(file, chunk_length)
    decisions = make_decision(preliminary_decisions)

    # Times are chunk lengths, not time series, so need to be summed
    plt.plot(np.cumsum(times), decision_to_num(decisions), label=label)
    plt.yticks(list(enum.values()), list(enum.keys()))
    plt.title(f"{title}, {chunk_length} second chunk")
    plt.legend()


def generate_video(audio_file: pathlib.Path, output_file: pathlib.Path, chunk_length: float) -> None:
    colors = [BAJI, LOWJI]
    preliminary_decisions, times = get_decisions(audio_file, chunk_length)
    decisions = make_decision(preliminary_decisions)

    # Make image clips for each decision. Need to set both duration and start time
    # or they won't end up correct when concatenated
    clip_list = []
    start = 0.0
    for decision, time in zip(decisions, times, strict=True):
        img = Image.new("RGB", (100, 100), ImageColor.getrgb(colors[enum[decision] > 0]))
        image_clip = ImageClip(np.array(img))
        image_clip.duration = time
        image_clip.start = start
        clip_list.append(image_clip)
        start += time
    audio_clip = AudioFileClip(audio_file)

    # Combine clips, combine with audio track, set metadata
    video = CompositeVideoClip(clip_list)
    video = video.with_audio(audio_clip)
    video.duration = sum(times)
    video.fps = 30

    # Write the result to a file, need to set codec(s) or audio won't play in Discord
    video.write_videofile(
        output_file,
        codec="libx264",
        audio_codec="aac",
    )
