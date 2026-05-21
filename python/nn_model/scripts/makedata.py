import numpy as np
import scipy.io.wavfile as wav


def time_range_to_sample_range(
    start: float,
    end: float,
    sample_rate: int,
) -> tuple[int, int]:
    first = int(start * sample_rate)
    last = int(end * sample_rate)
    return (
        first,
        last,
    )


def audio_meta(path: str) -> tuple[int, int]:
    sample_rate, audio_data = wav.read(path)
    return sample_rate, len(audio_data)


def load_noise(audio: str):
    sample_rate, samples = audio_meta(audio)
    out_scores: np.ndarray = np.zeros((samples, 4), dtype=np.byte)
    out_scores[:, 3].fill(1)
    return out_scores


# Convert a list of labels exported from Audacity to a 2d array of scores (per each audio sample)
def scores_from_labels(audio: str, labels: str):
    label_map: dict[str, list[tuple[float, float]]] = {
        "bn": [],  # baji/lowji noise (any)
        "b": [],  # baji voice features
        "l": [],  # lowji voice features
        "en": [],  # additional label for environment noises
    }
    with open(labels) as file:
        for line in file.readlines():
            [start, end, label] = line.strip().split("\t")
            label_map[label].append((float(start), float(end)))

    sample_rate, samples = audio_meta(audio)

    label_scores: dict[str, np.ndarray] = {}
    for label, time_ranges in label_map.items():
        scores = np.zeros(samples, dtype=np.byte)
        for f, t in time_ranges:
            (f_s, t_s) = time_range_to_sample_range(f, t, sample_rate)
            scores[f_s:t_s] = 1
        label_scores[label] = scores

    out_scores: np.ndarray = np.zeros((samples, 2), dtype=np.byte)
    out_scores[:, 0] = label_scores["bn"] & label_scores["b"]  # baji sound
    out_scores[:, 1] = label_scores["bn"] & label_scores["l"]  # lowji sound
    out_scores[:, 2] = label_scores["bn"] - np.maximum(
        out_scores[:, 0], out_scores[:, 1]
    )  # baji/lowji noises (only if there's no voice features)
    out_scores[:, 3] = np.maximum(
        (np.ones(samples) - label_scores["bn"]), label_scores["en"]
    )  # environment noises (only if there's no baji/lowji features)

    return out_scores


def smooth_scores(scores: np.ndarray, shift: int) -> np.ndarray:
    scores = scores.astype(np.float16)  # using fp16 for less memory usage
    shifted = scores[:-shift, :].copy()
    shifted[:, :] *= scores[shift:, :]  # masking shifted scores
    scores[shift:, :] += shifted[:, :]  # making "triangle"
    scores[:, :] *= 0.5  # normalization
    return scores


# Makes 2 raw files for audio data and scores
def convert_noise(path: str):
    sample_rate, audio_data = wav.read(path + ".wav")
    np.memmap(
        path + ".audio", dtype=audio_data.dtype, mode="w+", shape=audio_data.shape
    )[:] = audio_data[:]

    scores = load_noise(path + ".wav")
    np.memmap(path + ".scores", dtype=scores.dtype, mode="w+", shape=scores.shape)[
        :, :
    ] = scores[:, :]


# Makes 2 raw files for audio data and scores
def convert_labeled(path: str):
    sample_rate, audio_data = wav.read(path + ".wav")
    np.memmap(
        path + ".audio", dtype=audio_data.dtype, mode="w+", shape=audio_data.shape
    )[:] = audio_data[:]

    scores = scores_from_labels(path + ".wav", path + ".txt")
    np.memmap(path + ".scores", dtype=scores.dtype, mode="w+", shape=scores.shape)[
        :, :
    ] = scores[:, :]


if __name__ == "__main__":
    convert_labeled("baji_speech")
    convert_noise("only_noise")
