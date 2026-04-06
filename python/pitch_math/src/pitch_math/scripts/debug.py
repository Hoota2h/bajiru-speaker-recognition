"""Audio diagnostics monitor for configuration tuning.

Usage::

    pitch-debug [--device INDEX] [--vad-threshold THRESHOLD] [--verbose]

Displays real-time measurements — pitch, formants F1/F2, HNR, speech coverage,
and signal level — to help tune :data:`~pitch_math.config.VAD_THRESHOLD`,
:data:`~pitch_math.config.WINDOW_DURATION_MS`, and related constants.
No trained model is required.

Press Q to quit.
"""

import argparse
import collections
import logging
import sys
import threading
import time
from typing import ClassVar

import numpy as np
import sounddevice as sd
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Static

from pitch_math import __version__
from pitch_math.config import SAMPLE_RATE, STREAM_BLOCK_MS, VAD_THRESHOLD, WINDOW_DURATION_MS
from pitch_math.features import AudioDiagnostics, compute_diagnostics

logger = logging.getLogger(__name__)

_BAR_LEN = 20
_BAR_FILLED = "█"
_BAR_EMPTY = "░"


def _bar(value: float, max_value: float) -> str:
    """Render a fixed-width bar scaled to *max_value*.

    Args:
        value: Current measurement value.
        max_value: Value that corresponds to a full bar.

    Returns:
        A ``_BAR_LEN``-character string of filled and empty block characters.

    """
    filled = round(min(value / max_value, 1.0) * _BAR_LEN)
    return _BAR_FILLED * filled + _BAR_EMPTY * (_BAR_LEN - filled)


def _hint(diag: AudioDiagnostics) -> str:
    """Return a Textual-markup configuration hint based on the diagnostics.

    Args:
        diag: The most recent :class:`~pitch_math.features.AudioDiagnostics`.

    Returns:
        A string with Textual markup (``[yellow]``/``[green]``) suitable for
        passing directly to :meth:`textual.widgets.Static.update`.

    """
    if diag.rms < 0.002:
        return "[yellow]Low input level — check microphone or boost gain[/yellow]"
    if diag.speech_ratio == 0.0:
        return "[yellow]No speech detected — try lowering VAD_THRESHOLD[/yellow]"
    if diag.speech_ratio < 0.25:
        return "[yellow]Low speech coverage — consider lowering VAD_THRESHOLD[/yellow]"
    if diag.speech_ratio > 0.9:
        return "[yellow]High speech coverage — consider raising VAD_THRESHOLD[/yellow]"
    if 0.0 < diag.hnr_db < 5.0:
        return "[yellow]Low HNR — background noise may be affecting results[/yellow]"
    return "[green]Signal looks clean[/green]"


class _AudioMonitor:
    """Fills a ring buffer from the microphone and exposes :class:`~pitch_math.features.AudioDiagnostics` snapshots.

    Args:
        sample_rate: Audio sample rate in Hz.
        window_duration_ms: Length of the analysis window in milliseconds.
        stream_block_ms: sounddevice InputStream block size in milliseconds.
        vad_threshold: Silero VAD speech-probability threshold.
        device: PortAudio input device index or name.

    """

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        window_duration_ms: int = WINDOW_DURATION_MS,
        stream_block_ms: int = STREAM_BLOCK_MS,
        vad_threshold: float = VAD_THRESHOLD,
        device: int | str | None = None,
    ) -> None:
        """Initialise the monitor with audio settings."""
        self._sample_rate = sample_rate
        self._vad_threshold = vad_threshold
        self._stream_block_ms = stream_block_ms
        self._device = device
        self._running = False

        window_samples = int(sample_rate * window_duration_ms / 1000)
        self._ring: collections.deque[float] = collections.deque(maxlen=window_samples)
        self._lock = threading.Lock()
        self._latest: AudioDiagnostics | None = None
        self._stream: sd.InputStream | None = None
        self._thread: threading.Thread | None = None

    def _callback(self, indata: np.ndarray, _frames: int, _time_info: object, status: sd.CallbackFlags) -> None:
        if status:
            logger.debug("Stream status: %s", status)
        with self._lock:
            self._ring.extend(indata[:, 0])

    def _loop(self) -> None:
        min_samples = int(self._sample_rate * 0.3)
        while self._running:
            time.sleep(0.2)
            with self._lock:
                if len(self._ring) < min_samples:
                    continue
                audio = np.array(self._ring, dtype=np.float32)
            self._latest = compute_diagnostics(audio, self._sample_rate, vad_threshold=self._vad_threshold)

    def start(self) -> None:
        """Open the microphone stream and start the polling thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=int(self._sample_rate * self._stream_block_ms / 1000),
            callback=self._callback,
            device=self._device,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop the polling thread and close the microphone stream."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    @property
    def latest(self) -> AudioDiagnostics | None:
        """Most recent diagnostics snapshot, or ``None`` if not yet available."""
        return self._latest


class _DiagnosticsPanel(Static):
    """Displays live audio diagnostics and configuration hints."""

    DEFAULT_CSS = """
    _DiagnosticsPanel {
        height: 13;
        padding: 0 2;
    }
    """

    def update_diagnostics(self, diag: AudioDiagnostics) -> None:
        """Refresh the panel from a diagnostics snapshot.

        Args:
            diag: The most recent :class:`~pitch_math.features.AudioDiagnostics`.

        """
        speech_pct = diag.speech_ratio * 100
        pitch_str = f"{diag.pitch_hz:.0f} Hz" if diag.pitch_hz > 0 else "N/A    "
        f1_str = f"{diag.f1_hz:.0f} Hz" if diag.f1_hz > 0 else "N/A    "
        f2_str = f"{diag.f2_hz:.0f} Hz" if diag.f2_hz > 0 else "N/A    "
        hnr_str = f"{diag.hnr_db:.1f} dB" if diag.hnr_db > 0 else "N/A    "

        self.update(
            f"Speech  [{_bar(diag.speech_ratio, 1.0)}] {speech_pct:4.0f}%\n"
            f"Level   [{_bar(diag.rms, 0.1)}] {diag.rms:.4f}\n"
            f"\n"
            f"Pitch   {pitch_str}\n"
            f"F1      {f1_str}\n"
            f"F2      {f2_str}\n"
            f"HNR     {hnr_str}\n"
            f"\n"
            f"{_hint(diag)}",
        )


class DiagnosticsApp(App[None]):
    """Textual TUI application for live audio diagnostics monitoring.

    Args:
        device: PortAudio input device index or name. ``None`` uses the
            system default.
        vad_threshold: Silero VAD speech-probability threshold to use for
            this session, overriding :data:`~pitch_math.config.VAD_THRESHOLD`.

    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("q", "quit", "Quit")]

    def __init__(
        self,
        *,
        device: int | str | None = None,
        vad_threshold: float = VAD_THRESHOLD,
        **kwargs: object,
    ) -> None:
        """Initialise the app and create the audio monitor."""
        super().__init__(**kwargs)
        self._monitor = _AudioMonitor(device=device, vad_threshold=vad_threshold)

    def compose(self) -> ComposeResult:
        """Build the widget tree."""
        yield Header()
        yield _DiagnosticsPanel("[dim]Listening…[/dim]")
        yield Footer()

    def on_mount(self) -> None:
        """Start the audio monitor and begin polling for snapshots."""
        self.title = f"Audio Diagnostics Monitor — v{__version__}"
        self._monitor.start()
        self.set_interval(0.2, self._poll)

    def on_unmount(self) -> None:
        """Stop the audio monitor on exit."""
        self._monitor.stop()

    def _poll(self) -> None:
        """Refresh the panel from the latest diagnostics snapshot."""
        diag = self._monitor.latest
        if diag is None:
            return
        self.query_one(_DiagnosticsPanel).update_diagnostics(diag)


def main() -> None:
    """Entry point for the ``pitch-debug`` command."""
    parser = argparse.ArgumentParser(
        prog="pitch-debug",
        description=(
            "Live audio diagnostics monitor for tuning VAD_THRESHOLD, "
            "WINDOW_DURATION_MS, and other configuration values. "
            "No trained model is required."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="INDEX",
        help="PortAudio input device index or name (default: system default).",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=VAD_THRESHOLD,
        metavar="THRESHOLD",
        help=f"Override VAD_THRESHOLD for this session (default: {VAD_THRESHOLD}).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging (written to stderr).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        stream=sys.stderr,
    )

    device: int | str | None = int(args.device) if args.device is not None and args.device.isdigit() else args.device

    app = DiagnosticsApp(device=device, vad_threshold=args.vad_threshold)
    app.run()
