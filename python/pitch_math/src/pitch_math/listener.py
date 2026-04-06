"""Background audio listener for continuous real-time voice classification."""

import collections
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from sklearn.pipeline import Pipeline

from pitch_math.classifier import predict
from pitch_math.config import SAMPLE_RATE, STREAM_BLOCK_MS, WINDOW_DURATION_MS
from pitch_math.features import AudioDiagnostics, compute_diagnostics, extract_features

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Snapshot produced by a single classification cycle.

    Attributes:
        label: Predicted class index (e.g. 0 = LOW, 1 = HIGH).
        proba: Per-class probability array, shape ``(n_classes,)``.
        pitch_hz: Estimated mean F0 in Hz. ``0.0`` when no voiced frames found.
        elapsed_ms: Wall-clock time for the full classify iteration in ms.

    """

    label: int
    proba: np.ndarray
    pitch_hz: float
    elapsed_ms: float
    diagnostics: AudioDiagnostics | None = None

    @property
    def confidence(self) -> float:
        """Probability of the predicted class in [0.0, 1.0]."""
        return float(self.proba[self.label])


class VoiceListener:
    """Continuously classifies microphone audio in a rolling window.

    A :class:`sounddevice.InputStream` callback fills a ring buffer, and a
    background thread wakes up every *poll_interval_s* seconds to extract
    features and run the classifier.  Retrieve the most recent result at any
    time via :attr:`result`.

    Args:
        clf: Fitted classification pipeline.
        sample_rate: Audio sample rate in Hz.
        window_duration_ms: Length of the classification window in ms.
        stream_block_ms: sounddevice InputStream block size in ms.
        poll_interval_s: How often the classification thread wakes up.
        device: PortAudio input device index or name. ``None`` uses the system
            default.

    """

    def __init__(
        self,
        clf: Pipeline,
        *,
        sample_rate: int = SAMPLE_RATE,
        window_duration_ms: int = WINDOW_DURATION_MS,
        stream_block_ms: int = STREAM_BLOCK_MS,
        poll_interval_s: float = 0.15,
        device: int | str | None = None,
    ) -> None:
        """Initialise the listener with a fitted classifier and audio settings."""
        self._clf = clf
        self._sample_rate = sample_rate
        self._stream_block_ms = stream_block_ms
        self._poll_interval_s = poll_interval_s
        self._device = device
        self._running = False

        window_samples = int(sample_rate * window_duration_ms / 1000)
        self._ring: collections.deque[float] = collections.deque(maxlen=window_samples)
        self._lock = threading.Lock()
        self._result: ClassificationResult | None = None
        self._thread: threading.Thread | None = None
        self._stream: sd.InputStream | None = None

    def _audio_callback(self, indata: np.ndarray, _frames: int, _time_info: object, status: sd.CallbackFlags) -> None:
        if status:
            logger.debug("Stream status: %s", status)
        with self._lock:
            self._ring.extend(indata[:, 0])

    def _classify_loop(self) -> None:
        min_samples = int(self._sample_rate * 0.3)
        while self._running:
            time.sleep(self._poll_interval_s)
            t_start = time.perf_counter()

            with self._lock:
                if len(self._ring) < min_samples:
                    continue
                audio = np.array(self._ring, dtype=np.float32)

            features = extract_features(audio, self._sample_rate)
            if features is None:
                continue

            label, proba = predict(self._clf, features)
            diagnostics = compute_diagnostics(audio, self._sample_rate)

            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logger.debug(
                "label=%s confidence=%.1f%% pitch=%.1fHz elapsed=%.1fms",
                label,
                float(proba[label]) * 100,
                diagnostics.pitch_hz,
                elapsed_ms,
            )

            self._result = ClassificationResult(
                label=label,
                proba=proba,
                pitch_hz=diagnostics.pitch_hz,
                elapsed_ms=elapsed_ms,
                diagnostics=diagnostics,
            )

    def start(self) -> None:
        """Open the microphone stream and start the classification thread."""
        self._running = True
        self._thread = threading.Thread(target=self._classify_loop, daemon=True)
        self._thread.start()
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=int(self._sample_rate * self._stream_block_ms / 1000),
            callback=self._audio_callback,
            device=self._device,
        )
        self._stream.start()
        logger.info(
            "Listener started — device=%s sample_rate=%d",
            self._device,
            self._sample_rate,
        )

    def stop(self) -> None:
        """Stop the classification thread and close the microphone stream."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        logger.info("Listener stopped")

    @property
    def result(self) -> ClassificationResult | None:
        """Most recent classification result, or ``None`` if not yet available."""
        return self._result

    @property
    def device(self) -> int | str | None:
        """The PortAudio device passed at construction time."""
        return self._device
