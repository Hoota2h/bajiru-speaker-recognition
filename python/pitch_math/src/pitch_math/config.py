"""Global constants for audio processing and model configuration."""

SAMPLE_RATE: int = 16000
"""Sample rate in Hz. Silero VAD requires either 8000 or 16000."""

STREAM_BLOCK_MS: int = 30
"""Audio stream block size in milliseconds used by the sounddevice InputStream."""

WINDOW_DURATION_MS: int = 750
"""Length of the rolling classification window in milliseconds."""

N_MFCC: int = 20
"""Number of Mel-frequency cepstral coefficients to compute."""

VAD_THRESHOLD: float = 0.5
"""Silero VAD speech-probability threshold (0.0-1.0). Higher values are stricter."""

DEFAULT_MODEL_PATH: str = "voice_model.pkl"
"""Default filesystem path used when saving or loading the trained model."""

LABELS: dict[int, str] = {0: "LOW VOICE", 1: "HIGH VOICE"}
"""Human-readable class names keyed by integer label index."""
