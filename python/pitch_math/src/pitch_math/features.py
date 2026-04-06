"""Pure, stateless feature extraction for voice classification.

All functions are free of side effects and take only plain values as inputs,
making them straightforward to test and reuse across scripts.
"""

import functools
import logging
import time
from dataclasses import dataclass

import librosa
import numpy as np
import parselmouth
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from pitch_math.config import N_MFCC, VAD_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class AudioDiagnostics:
    """Human-interpretable measurements extracted from an audio window.

    Attributes:
        pitch_hz: Mean fundamental frequency in Hz across voiced frames.
            ``0.0`` when no voiced frames were detected.
        f1_hz: First formant frequency mean in Hz. ``0.0`` on failure.
        f2_hz: Second formant frequency mean in Hz. ``0.0`` on failure.
        hnr_db: Mean harmonics-to-noise ratio in dB. ``0.0`` on failure.
        speech_ratio: Proportion of the window classified as speech (0.0-1.0).
            ``0.0`` when VAD detects no speech.
        rms: Root-mean-square energy of the full (pre-VAD) window.

    """

    pitch_hz: float
    f1_hz: float
    f2_hz: float
    hnr_db: float
    speech_ratio: float
    rms: float


@functools.cache
def _vad_model() -> object:
    """Return the Silero VAD model, loading it on the first call.

    The result is cached for the lifetime of the process so the model is only
    loaded once regardless of how many times ``filter_speech_frames`` or
    ``extract_features`` are called.

    Returns:
        The loaded Silero VAD torch model.

    """
    logger.debug("Loading Silero VAD model…")
    return load_silero_vad()


def filter_speech_frames(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold: float = VAD_THRESHOLD,
) -> np.ndarray | None:
    """Retain only speech portions of an audio array using Silero VAD.

    Args:
        audio: Float32 mono audio array.
        sample_rate: Sample rate of *audio* in Hz. Silero VAD supports 8000
            and 16000 Hz.
        threshold: Speech-probability threshold in [0.0, 1.0]. Higher values
            require more confidence before a region is labelled as speech.

    Returns:
        Concatenated speech-only audio array, or ``None`` if no speech was
        detected or the total speech duration is under 150 ms.

    """
    model = _vad_model()
    tensor = torch.from_numpy(audio)
    timestamps = get_speech_timestamps(
        tensor,
        model,
        sampling_rate=sample_rate,
        threshold=threshold,
    )
    if not timestamps:
        return None

    segments = [audio[ts["start"] : ts["end"]] for ts in timestamps]
    speech = np.concatenate(segments)

    if len(speech) < sample_rate * 0.15:  # Require at least 150 ms
        return None

    return speech


def _extract_praat_features(speech: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract formant and HNR features from a speech array via Praat.

    Uses the Burg LPC method for formant tracking and the cross correlation
    method for harmonics to noise ratio.  Both measures are amplitude
    independent, so RMS normalised input is fine.

    Args:
        speech: Float32 mono speech array.
        sample_rate: Sample rate in Hz.

    Returns:
        1-D array of 6 values:
        ``[F1_mean, F1_std, F2_mean, F2_std, HNR_mean, HNR_std]``.
        Returns zeros on any failure so the feature vector length stays fixed.

    """
    fallback = np.zeros(6)
    try:
        sound = parselmouth.Sound(speech.astype(np.float64), sample_rate)

        # Formants F1 and F2, vocal tract resonances shift between registers
        formant = sound.to_formant_burg(
            max_number_of_formants=5,
            maximum_formant=5500.0,
        )
        times = formant.xs()
        f1 = np.array([formant.get_value_at_time(1, t) for t in times])
        f2 = np.array([formant.get_value_at_time(2, t) for t in times])
        f1 = f1[np.isfinite(f1)]
        f2 = f2[np.isfinite(f2)]
        if len(f1) == 0 or len(f2) == 0:
            return fallback

        # HNR, chest/low voice has more complete glottal closure and therefore
        # cleaner harmonics; breathy/high voice produces a lower HNR.
        harmonicity = sound.to_harmonicity_cc()
        hnr = harmonicity.values.squeeze()
        hnr = hnr[np.isfinite(hnr) & (hnr > -190)]  # -200 dB = undefined frame
        if len(hnr) == 0:
            hnr = np.array([0.0])

        return np.array(
            [
                np.mean(f1),
                np.std(f1),
                np.mean(f2),
                np.std(f2),
                np.mean(hnr),
                np.std(hnr),
            ],
        )
    except Exception:
        logger.debug("Praat feature extraction failed", exc_info=True)
        return fallback


def compute_pitch(audio: np.ndarray, sample_rate: int) -> float:
    """Estimate the mean fundamental frequency across voiced frames.

    Args:
        audio: Float32 mono audio array.
        sample_rate: Sample rate in Hz.

    Returns:
        Mean F0 in Hz across voiced frames, or ``0.0`` if no voiced frames
        were found.

    """
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
    )
    f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
    return float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0


def compute_diagnostics(
    audio: np.ndarray,
    sample_rate: int,
    *,
    vad_threshold: float = VAD_THRESHOLD,
) -> AudioDiagnostics:
    """Compute a lightweight set of diagnostic measurements from an audio window.

    Unlike :func:`extract_features`, this function returns human-interpretable
    values rather than a flat feature vector.  It is intended for display and
    configuration tuning, not classification.

    Args:
        audio: Float32 mono audio array.
        sample_rate: Sample rate in Hz.
        vad_threshold: Silero VAD speech-probability threshold (0.0-1.0).

    Returns:
        :class:`AudioDiagnostics` populated from the audio segment.  All
        fields that could not be computed default to ``0.0``.

    """
    rms = float(np.sqrt(np.mean(audio**2)))
    speech = filter_speech_frames(audio, sample_rate, threshold=vad_threshold)

    if speech is None:
        return AudioDiagnostics(
            pitch_hz=0.0,
            f1_hz=0.0,
            f2_hz=0.0,
            hnr_db=0.0,
            speech_ratio=0.0,
            rms=rms,
        )

    speech_ratio = len(speech) / max(len(audio), 1)
    pitch_hz = compute_pitch(speech, sample_rate)
    praat = _extract_praat_features(speech, sample_rate)

    return AudioDiagnostics(
        pitch_hz=pitch_hz,
        f1_hz=float(praat[0]),
        f2_hz=float(praat[2]),
        hnr_db=float(praat[4]),
        speech_ratio=speech_ratio,
        rms=rms,
    )


def extract_features(
    audio: np.ndarray,
    sample_rate: int,
    *,
    vad_threshold: float = VAD_THRESHOLD,
    n_mfcc: int = N_MFCC,
) -> np.ndarray | None:
    """Extract a fixed-length feature vector from an audio segment.

    The returned vector is the concatenation of:

    * MFCC means and standard deviations (``n_mfcc * 2`` values)
    * Delta-MFCC means (``n_mfcc`` values, captures temporal dynamics)
    * Pitch statistics: mean, std, median, P25, P75 (5 values)
    * Spectral centroid, rolloff, bandwidth, and ZCR - mean + std each
      (8 values)
    * Spectral contrast means and standard deviations across sub-bands
      (14 values at the default n_fft)
    * Formant F1/F2 statistics and HNR via Praat (6 values)

    Args:
        audio: Float32 mono audio array.
        sample_rate: Sample rate in Hz.
        vad_threshold: Silero VAD speech-probability threshold (0.0-1.0).
        n_mfcc: Number of MFCC coefficients.

    Returns:
        1-D float64 feature vector, or ``None`` if the segment contains
        insufficient speech content.

    """
    if len(audio) < sample_rate * 0.1:
        return None

    t_start = time.perf_counter()

    speech = filter_speech_frames(
        audio,
        sample_rate,
        threshold=vad_threshold,
    )
    if speech is None:
        return None

    # Isolate the harmonic component to reduce percussive and noise influence
    speech = librosa.effects.harmonic(speech, margin=3.0)

    # RMS normalise to decouple loudness from the feature space
    rms = float(np.sqrt(np.mean(speech**2)))
    if rms > 1e-6:
        speech = speech / rms

    # Largest power of two <= signal length avoids a librosa warning
    n_fft = min(2048, 2 ** int(np.floor(np.log2(len(speech)))))

    # MFCCs
    mfccs = librosa.feature.mfcc(y=speech, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    # Delta MFCCs, width must be odd and <= n_frames
    n_frames = mfccs.shape[1]
    if n_frames >= 3:
        delta_width = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
        delta_mfcc = librosa.feature.delta(mfccs, width=delta_width)
    else:
        delta_mfcc = np.zeros_like(mfccs)
    delta_mean = np.mean(delta_mfcc, axis=1)

    # Fundamental frequency (F0), discriminator for low vs high voice
    f0, voiced_flag, _ = librosa.pyin(
        speech,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
    )
    f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
    if len(f0_voiced) == 0:
        f0_voiced = np.array([0.0])

    pitch_features = np.array(
        [
            np.mean(f0_voiced),
            np.std(f0_voiced),
            np.median(f0_voiced),
            float(np.percentile(f0_voiced, 25)),
            float(np.percentile(f0_voiced, 75)),
        ],
    )

    # Spectral shape features
    centroid = librosa.feature.spectral_centroid(y=speech, sr=sample_rate, n_fft=n_fft)
    rolloff = librosa.feature.spectral_rolloff(y=speech, sr=sample_rate, n_fft=n_fft)
    bandwidth = librosa.feature.spectral_bandwidth(y=speech, sr=sample_rate, n_fft=n_fft)
    zcr = librosa.feature.zero_crossing_rate(speech)

    spectral_features = np.array(
        [
            np.mean(centroid),
            np.std(centroid),
            np.mean(rolloff),
            np.std(rolloff),
            np.mean(bandwidth),
            np.std(bandwidth),
            np.mean(zcr),
            np.std(zcr),
        ],
    )

    # Spectral contrast measures the tonal peak-to-valley ratio per subband
    contrast = librosa.feature.spectral_contrast(y=speech, sr=sample_rate, n_fft=n_fft)
    contrast_features = np.concatenate(
        [
            np.mean(contrast, axis=1),
            np.std(contrast, axis=1),
        ],
    )

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.debug("Feature extraction completed in %.1f ms", elapsed_ms)

    # Formant and HNR features via Praat
    praat_features = _extract_praat_features(speech, sample_rate)

    return np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            delta_mean,
            pitch_features,
            spectral_features,
            contrast_features,
            praat_features,
        ],
    )
