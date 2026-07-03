import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import scipy.io.wavfile as wav

PROGRESS_DISPLAY_INTERVAL = 0.5


# Asymmetric triangular kernel
def triangular_kernel(front_samples: int, rear_samples: int) -> np.ndarray:
    front_part = np.linspace(1.0 / front_samples, 1.0, front_samples, endpoint=True)
    rear_part = np.linspace(
        (front_part[-1] * (rear_samples - 1)) / rear_samples,
        1.0 / rear_samples,
        rear_samples,
        endpoint=True,
    )[::-1]
    kern = np.concatenate([front_part, rear_part])
    kern = kern.astype(np.float64)
    kern /= kern.sum()
    return kern


def labels_to_scores(
    input_path: str,
    output_path: str,
    n_samples: int,
    n_scores: int,
    front_samples: int,
    rear_samples: int,
    chunk_size: int,
):
    """Convert labels to scores used for model training, and smoothes them using triangular kernel

    Args:
        input_path (str): Path to 2d array with labels
        output_path (str): Path to 2d array with scores
        n_samples (int): Total number of samples in the audio file
        n_scores (int): Number of label/score types
        front_samples (int): Number of front samples for triangular kernel
        rear_samples (int): Number of rear samples for triangular kernel
        chunk_size (int): The size of the chunk used for batching

    """
    inp = np.memmap(input_path, dtype=np.uint8, mode="r+", shape=(n_samples, n_scores))
    out = np.memmap(
        output_path, dtype=np.float16, mode="w+", shape=(n_samples, n_scores)
    )

    kern = triangular_kernel(front_samples, rear_samples)
    klen = len(kern)
    overlap = klen - 1

    def process_chunk(chunk_start: int, chunk_end: int):
        fetch_start = max(0, chunk_start - overlap)
        fetch_end = min(n_samples, chunk_end + overlap)
        block = np.array(inp[fetch_start:fetch_end], dtype=np.float64)
        block_len = chunk_end - chunk_start
        out_block = np.empty((block_len, n_scores), dtype=np.float64)

        def conv_channel(ch_idx: int):
            col = block[:, ch_idx]
            conv_full = np.convolve(col, kern, mode="full")
            start_idx = chunk_start - fetch_start
            end_idx = start_idx + block_len
            return conv_full[start_idx:end_idx]

        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 1, n_scores)) as exe:
            futures = {exe.submit(conv_channel, ch): ch for ch in range(n_scores)}
            for f in as_completed(futures):
                ch = futures[f]
                out_block[:, ch] = f.result()
        out[chunk_start:chunk_end] = out_block.astype(np.float16)

    starts = list(range(0, n_samples, chunk_size))
    chunks = [(s, min(n_samples, s + chunk_size)) for s in starts]

    processed = 0
    start_time = time.time()
    last_report = start_time
    for i, (s, e) in enumerate(chunks):
        process_chunk(s, e)
        processed += e - s
        now = time.time()
        if now - last_report >= PROGRESS_DISPLAY_INTERVAL or processed == n_samples:
            elapsed = now - start_time
            frac = processed / n_samples
            eta = (elapsed / frac) - elapsed if frac > 0 else float("inf")
            print(
                f"Chunk {i + 1}/{len(chunks)}: processed {processed}/{n_samples} samples "
                f"({frac * 100:.2f}%), elapsed {elapsed:.1f}s, ETA {eta:.1f}s"
            )
            last_report = now

    out.flush()


def audio_meta(path: str) -> tuple[int, int]:
    sample_rate, audio_data = wav.read(path)
    return sample_rate, len(audio_data)


def time_range_to_sample_range(
    start: float,
    end: float,
    sample_rate: int,
) -> tuple[int, int]:
    first = int(start * sample_rate)
    last = int(end * sample_rate)
    return (first, last)


def map_labels(
    labels_path: str, sample_rate: int, n_samples: int, output_path: str
) -> int:
    """Map labels from the text file to 2d array

    Args:
        labels_path (str): Labels text file, exported from Audacity
        sample_rate (int): Audio sample rate
        n_samples (int): Total number of samples in the audio file
        output_path (str): The output file for labels. 2d array of bytes ([n_labels, n_samples]), where label can have value 1 or 0.

    Returns:
        int: The number of label types

    """
    # Train label list:
    # b - baji
    # r - ru
    # n - baji/ru noise
    # e - env. noise
    # i - instrumental
    # s - singing
    # v - other voice
    # y - yelling
    # w - whisper
    label_map: dict[str, list[tuple[float, float]]] = {
        "p": [],  # All the train labels (except the noise) are multiplied by the peak label
        "b": [],
        "r": [],
        "n": [],
        "e": [],
        "i": [],
        "s": [],
        "v": [],
        "y": [],
        "w": [],
    }
    with open(labels_path) as file:
        for line in file:
            [start, end, label] = line.strip().split("\t")
            label_map[label].append((float(start), float(end)))

    label_samples: dict[str, np.ndarray] = {}
    for label, time_ranges in label_map.items():
        l_samples = np.zeros(n_samples, dtype=np.uint8)
        for f, t in time_ranges:
            (f_s, t_s) = time_range_to_sample_range(f, t, sample_rate)
            if f_s >= n_samples:
                continue
            t_s = min(t_s, n_samples - 1)
            l_samples[f_s:t_s] = 1
        label_samples[label] = l_samples

    n_scores = 5
    out = np.memmap(output_path, dtype=np.uint8, mode="w+", shape=(n_samples, n_scores))

    out[:, 0] = label_samples["p"] * label_samples["b"]  # baji voice
    out[:, 1] = label_samples["p"] * label_samples["r"]  # ru voice
    out[:, 2] = np.maximum(
        (np.ones((n_samples,), dtype=np.uint8) - np.maximum(out[:, 0], out[:, 1])),
        label_samples["e"],
    )  # env noise
    out[:, 3] = label_samples["p"] * label_samples["i"]  # instrumentals
    out[:, 4] = label_samples["s"] * np.maximum(out[:, 0], out[:, 1])  # singing

    out.flush()
    return n_scores


# Makes scores file
def convert_dataset(labels_path: str, audio_path: str, output_path: str):
    sample_rate, samples = audio_meta(audio_path)

    if os.path.exists("tmp.labels"):
        os.remove("tmp.labels")
    n_scores = map_labels(labels_path, sample_rate, samples, "tmp.labels")
    labels_to_scores(
        "tmp.labels", output_path, samples, n_scores, 256, 1024 * 2, 1000000
    )
    os.remove("tmp.labels")


# Makes raw audio file
def convert_audio(input_path: str, output_path: str, target_samples: int | None = None):
    _sample_rate, audio_data = wav.read(input_path)
    if target_samples is None:
        target_samples = len(audio_data)
    file = np.memmap(
        output_path, dtype=audio_data.dtype, mode="w+", shape=(target_samples,)
    )
    src_samples = audio_data.shape[0]
    if src_samples > target_samples:
        file[:] = audio_data[:target_samples]
    else:
        file[:src_samples] = audio_data[:]


# Makes 2 raw files for audio data and scores
def convert_labeled(base_path: str):
    convert_dataset(
        base_path + ".txt",
        base_path + ".wav",
        base_path + ".scores",
    )
    convert_audio(base_path + ".wav", base_path + ".audio")


if __name__ == "__main__":
    convert_labeled("baji_speech")  # base path to the .waw/.txt files
