"""Model construction, training, persistence, and inference utilities."""

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pitch_math.config import DEFAULT_MODEL_PATH

logger = logging.getLogger(__name__)


def build_classifier() -> Pipeline:
    """Construct an untrained SVM classification pipeline.

    Returns:
        A :class:`~sklearn.pipeline.Pipeline` containing a
        :class:`~sklearn.preprocessing.StandardScaler` followed by a
        radial-basis-function :class:`~sklearn.svm.SVC` with probability
        estimation enabled.

    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
        ],
    )


def train(
    low_features: list[np.ndarray],
    high_features: list[np.ndarray],
) -> Pipeline:
    """Fit a new classifier on pre-extracted feature vectors.

    Args:
        low_features: Feature vectors for the low-voice class (label 0).
        high_features: Feature vectors for the high-voice class (label 1).

    Returns:
        Fitted :class:`~sklearn.pipeline.Pipeline`.

    """
    x = low_features + high_features
    y = [0] * len(low_features) + [1] * len(high_features)
    clf = build_classifier()
    clf.fit(x, y)
    logger.info(
        "Model trained — %d low + %d high samples",
        len(low_features),
        len(high_features),
    )
    return clf


def predict(clf: Pipeline, features: np.ndarray) -> tuple[int, np.ndarray]:
    """Classify a single feature vector.

    Args:
        clf: A fitted classification pipeline.
        features: 1-D feature vector as returned by
            :func:`pitch.features.extract_features`.

    Returns:
        A ``(label_index, proba)`` tuple where *proba* is a float array of
        per-class probabilities summing to 1.0.

    """
    proba: np.ndarray = clf.predict_proba([features])[0]
    return int(np.argmax(proba)), proba


def save(clf: Pipeline, path: str | Path = DEFAULT_MODEL_PATH) -> None:
    """Serialise the fitted model to a pickle file.

    Args:
        clf: Fitted pipeline to persist.
        path: Destination file path. Parent directories must exist.

    """
    path = Path(path)
    with path.open("wb") as fh:
        pickle.dump(clf, fh)
    logger.info("Model saved to %s", path)


def load(path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline | None:
    """Deserialise a model from a pickle file.

    Args:
        path: Path to the pickle file produced by :func:`save`.

    Returns:
        The fitted pipeline, or ``None`` if the file does not exist.

    """
    path = Path(path)
    if not path.exists():
        logger.warning("Model file not found: %s", path)
        return None
    with path.open("rb") as fh:
        clf = pickle.load(fh)  # noqa: S301
    logger.info("Model loaded from %s", path)
    return clf
