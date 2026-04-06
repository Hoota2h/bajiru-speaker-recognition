"""Testing script: classify audio files and write structured results to a log.

Usage::

    pitch-test PATH [--model PATH] [--log FILE] [--verbose]

PATH may be a single audio file or a directory.  Each file is classified and
results are emitted via Python's logging system, making it easy to redirect
output to a file with ``--log``.
"""

import argparse
import logging
import sys
from pathlib import Path

from sklearn.pipeline import Pipeline

from pitch_math import __version__
from pitch_math.audio import load_file
from pitch_math.classifier import load, predict
from pitch_math.config import DEFAULT_MODEL_PATH, LABELS, SAMPLE_RATE
from pitch_math.features import compute_pitch, extract_features

logger = logging.getLogger(__name__)

_AUDIO_EXTENSIONS = ("*.mp3", "*.wav", "*.flac", "*.ogg", "*.m4a")


def _classify_file(path: Path, clf: Pipeline) -> None:
    """Classify a single audio file and emit results via the logger.

    Args:
        path: Path to the audio file to classify.
        clf: Fitted classification pipeline.

    """
    audio = load_file(str(path), sample_rate=SAMPLE_RATE)
    features = extract_features(audio, SAMPLE_RATE)

    if features is None:
        logger.warning("%s — insufficient speech detected, skipping", path.name)
        return

    label, proba = predict(clf, features)
    pitch_hz = compute_pitch(audio, SAMPLE_RATE)
    confidence = float(proba[label])

    logger.info(
        "%s — label=%-10s  confidence=%5.1f%%  pitch=%6.1f Hz  low_prob=%5.1f%%  high_prob=%5.1f%%",
        path.name,
        LABELS[label],
        confidence * 100,
        pitch_hz,
        float(proba[0]) * 100,
        float(proba[1]) * 100,
    )


def main() -> None:
    """Entry point for the ``pitch-test`` command."""
    parser = argparse.ArgumentParser(
        prog="pitch-test",
        description=(
            "Classify audio files and write structured results to the log.\n"
            "PATH may be a single file or a directory of audio files."
        ),
    )
    parser.add_argument(
        "path",
        metavar="PATH",
        help="Audio file or directory to classify.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        metavar="PATH",
        help=f"Path to the trained model (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--log",
        default=None,
        metavar="FILE",
        help="Write log output to this file instead of stdout.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    handlers: list[logging.Handler] = []
    if args.log:
        handlers.append(logging.FileHandler(args.log, encoding="utf-8"))
    else:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        handlers=handlers,
    )
    logger.info("pitch-test v%s", __version__)

    clf = load(args.model)
    if clf is None:
        logger.error(
            "No model found at %s. Train one first with: pitch-train LOW_DIR HIGH_DIR",
            args.model,
        )
        sys.exit(1)

    target = Path(args.path)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = []
        for pattern in _AUDIO_EXTENSIONS:
            files.extend(target.rglob(pattern))
        files.sort()
    else:
        logger.error("Path not found: %s", target)
        sys.exit(1)

    if not files:
        logger.error("No audio files found at %s", target)
        sys.exit(1)

    logger.info("Classifying %d file(s)…", len(files))
    for path in files:
        _classify_file(path, clf)
