"""Training script: build a voice model from two directories of audio files.

Usage::

    pitch-train LOW_DIR HIGH_DIR [--model PATH] [--verbose]

Both directories are scanned recursively for audio files (MP3, WAV, FLAC,
OGG, M4A).  Each file is split into overlapping windows that match the live
classification window size so training and inference share identical feature
distributions.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from pitch_math import __version__
from pitch_math.audio import load_file
from pitch_math.classifier import save, train
from pitch_math.config import DEFAULT_MODEL_PATH, SAMPLE_RATE, WINDOW_DURATION_MS
from pitch_math.features import extract_features

logger = logging.getLogger(__name__)

_AUDIO_EXTENSIONS = ("*.mp3", "*.wav", "*.flac", "*.ogg", "*.m4a")


def _collect_features(directory: Path, label: str) -> list[np.ndarray]:
    """Load every audio file in *directory* and return their feature vectors.

    Each file is sliced into overlapping windows (50% hop) whose length
    matches ``WINDOW_DURATION_MS`` so that training and inference observe the
    same feature distributions.

    Args:
        directory: Folder containing audio files for one voice class.
        label: Human-readable class name used only in log output.

    Returns:
        List of 1-D feature vectors.  Empty when no speech was detected in
        any file.

    """
    files: list[Path] = []
    for pattern in _AUDIO_EXTENSIONS:
        files.extend(directory.rglob(pattern))

    if not files:
        logger.error("No audio files found in %s", directory)
        return []

    window_samples = int(SAMPLE_RATE * WINDOW_DURATION_MS / 1000)
    hop_samples = window_samples // 2
    all_features: list[np.ndarray] = []

    for path in sorted(files):
        audio = load_file(str(path), sample_rate=SAMPLE_RATE)
        file_features: list[np.ndarray] = []
        for start in range(0, len(audio) - window_samples + 1, hop_samples):
            feat = extract_features(audio[start : start + window_samples], SAMPLE_RATE)
            if feat is not None:
                file_features.append(feat)
        all_features.extend(file_features)
        logger.info("%s | %s — %d windows extracted", label, path.name, len(file_features))

    logger.info("%s total: %d feature vectors from %d files", label, len(all_features), len(files))
    return all_features


def main() -> None:
    """Entry point for the ``pitch-train`` command."""
    parser = argparse.ArgumentParser(
        prog="pitch-train",
        description=(
            "Train a voice-mode classifier from two directories of audio files.\n"
            "LOW_DIR should contain recordings of the low voice; HIGH_DIR the high voice."
        ),
    )
    parser.add_argument(
        "low_dir",
        metavar="LOW_DIR",
        help="Directory of audio files for the LOW voice class.",
    )
    parser.add_argument(
        "high_dir",
        metavar="HIGH_DIR",
        help="Directory of audio files for the HIGH voice class.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        metavar="PATH",
        help=f"Output path for the trained model (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    logger.info("pitch-train v%s", __version__)

    low_dir = Path(args.low_dir)
    high_dir = Path(args.high_dir)

    for path, name in [(low_dir, "LOW_DIR"), (high_dir, "HIGH_DIR")]:
        if not path.is_dir():
            logger.error("%s is not a directory: %s", name, path)
            sys.exit(1)

    low_features = _collect_features(low_dir, "LOW")
    high_features = _collect_features(high_dir, "HIGH")

    if not low_features or not high_features:
        logger.error(
            "Insufficient training data (low=%d, high=%d). Aborting.",
            len(low_features),
            len(high_features),
        )
        sys.exit(1)

    clf = train(low_features, high_features)
    save(clf, args.model)
    logger.info("Model saved to %s", args.model)
