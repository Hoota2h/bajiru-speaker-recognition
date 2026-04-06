"""Live detection script with a Textual TUI.

Usage::

    pitch-live [--model PATH] [--device INDEX] [--verbose]

The app displays the current voice label in a coloured panel, a confidence
bar, pitch estimate, per-class probabilities, and the last parse time.
Press Q or Ctrl+C to quit.
"""

import argparse
import logging
import sys
from typing import TYPE_CHECKING, ClassVar

import sounddevice as sd
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Static

from pitch_math import __version__
from pitch_math.classifier import load
from pitch_math.config import DEFAULT_MODEL_PATH, LABELS
from pitch_math.listener import ClassificationResult, VoiceListener

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

_BAR_LEN = 20
_BAR_FILLED = "█"
_BAR_EMPTY = "░"


def _confidence_bar(confidence: float) -> str:
    """Render a fixed-width text progress bar for a confidence value.

    Args:
        confidence: Predicted-class probability in [0.0, 1.0].

    Returns:
        A string of filled and empty block characters of length ``_BAR_LEN``.

    """
    filled = round(confidence * _BAR_LEN)
    return _BAR_FILLED * filled + _BAR_EMPTY * (_BAR_LEN - filled)


class _LabelPanel(Static):
    """Large coloured panel that shows the current voice-mode label."""

    DEFAULT_CSS = """
    _LabelPanel {
        height: 5;
        content-align: center middle;
        text-align: center;
        text-style: bold;
        border: round $primary;
        margin-bottom: 1;
    }
    _LabelPanel.low {
        border: round #1e90ff;
        color: #1e90ff;
    }
    _LabelPanel.high {
        border: round #00cd00;
        color: #00cd00;
    }
    """

    def update_result(self, result: ClassificationResult) -> None:
        """Refresh the panel from the latest classification result.

        Args:
            result: The most recent :class:`~pitch.listener.ClassificationResult`.

        """
        self.remove_class("low", "high")
        self.add_class("low" if result.label == 0 else "high")
        self.update(LABELS[result.label])


class _StatsPanel(Static):
    """Displays confidence bar, pitch, per-class probabilities, and parse time."""

    DEFAULT_CSS = """
    _StatsPanel {
        height: 10;
        padding: 0 2;
    }
    """

    def update_result(self, result: ClassificationResult) -> None:
        """Refresh the stats from the latest classification result.

        Args:
            result: The most recent :class:`~pitch.listener.ClassificationResult`.

        """
        bar = _confidence_bar(result.confidence)
        pitch_str = f"{result.pitch_hz:.0f} Hz" if result.pitch_hz > 0 else "N/A    "
        low_pct = float(result.proba[0]) * 100
        high_pct = float(result.proba[1]) * 100

        d = result.diagnostics
        if d is not None:
            f1_str = f"{d.f1_hz:.0f} Hz" if d.f1_hz > 0 else "N/A    "
            f2_str = f"{d.f2_hz:.0f} Hz" if d.f2_hz > 0 else "N/A    "
            hnr_str = f"{d.hnr_db:.1f} dB" if d.hnr_db > 0 else "N/A    "
            speech_pct = d.speech_ratio * 100
            detail = (
                f"Pitch  {pitch_str}    HNR    {hnr_str}\n"
                f"F1     {f1_str}    F2     {f2_str}\n"
                f"Speech {speech_pct:4.0f}%\n"
            )
        else:
            detail = f"Pitch  {pitch_str}\n"

        self.update(
            f"Confidence  [{bar}] {result.confidence * 100:5.1f}%\n"
            + detail
            + f"\nLow  {low_pct:5.1f}%    High  {high_pct:5.1f}%\n\n"
            f"[dim]Parse time  {result.elapsed_ms:.0f} ms[/dim]",
        )


class LiveDetectApp(App[None]):
    """Textual TUI application for real-time voice mode classification.

    Args:
        clf: A fitted classification pipeline.
        device: PortAudio input device index or name. ``None`` uses the
            system default.

    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("q", "quit", "Quit")]

    def __init__(
        self,
        clf: "Pipeline",
        device: int | str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialise the app and create the :class:`~pitch.listener.VoiceListener`.

        Args:
            clf: Fitted classification pipeline.
            device: PortAudio input device index or name.
            **kwargs: Forwarded to :class:`textual.app.App`.

        """
        super().__init__(**kwargs)
        self._listener = VoiceListener(clf, device=device)

    def compose(self) -> ComposeResult:
        """Build the widget tree."""
        yield Header()
        yield _LabelPanel("[dim]Listening…[/dim]")
        yield _StatsPanel("")
        yield Footer()

    def on_mount(self) -> None:
        """Start the audio listener and begin polling for results."""
        self.title = f"Pitch Math - v{__version__}"
        self._listener.start()
        self.set_interval(0.2, self._poll)

    def on_unmount(self) -> None:
        """Stop the audio listener on exit."""
        self._listener.stop()

    def _poll(self) -> None:
        """Pull the latest result from the listener and refresh widgets."""
        result = self._listener.result
        if result is None:
            return
        self.query_one(_LabelPanel).update_result(result)
        self.query_one(_StatsPanel).update_result(result)


def _device_name(device: int | str | None) -> str:
    """Return a human-readable PortAudio device name.

    Args:
        device: Device index, name string, or ``None`` for the system default.

    Returns:
        Device name, or ``"default"`` if the lookup fails.

    """
    try:
        idx = device if device is not None else sd.default.device[0]
        return str(sd.query_devices(idx)["name"])
    except Exception:
        return "default"


def main() -> None:
    """Entry point for the ``pitch-live`` command."""
    parser = argparse.ArgumentParser(
        prog="pitch-live",
        description="Live voice-mode detection with a Textual TUI.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        metavar="PATH",
        help=f"Path to the trained model (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="INDEX",
        help="PortAudio input device index or name (default: system default).",
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
    logger.info("pitch-live v%s", __version__)

    clf = load(args.model)
    if clf is None:
        sys.stderr.write(
            f"No model found at {args.model}.\nTrain one first with:  pitch-train LOW_DIR HIGH_DIR\n",
        )
        sys.exit(1)

    device: int | str | None = int(args.device) if args.device is not None and args.device.isdigit() else args.device
    logger.info("Input device: %s", _device_name(device))

    app = LiveDetectApp(clf, device=device)
    app.run()
