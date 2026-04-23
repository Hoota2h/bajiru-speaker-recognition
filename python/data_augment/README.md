# data_augment

Offline audio data augmentation for baji vs. lowji expansion.

Combinatorial: each chain entry is `(name, config)`, config values are scalars (1 mode) or lists (N modes), and `augment(x)` returns the cartesian product across every dimension of every entry. Variant order follows the spec. Six of the seven augmentations are pure functions of `(signal, config)`; only `noise` draws RNG — pass `seed=<int>` if you care about reproducibility there.

## Install

```bash
uv sync
uv pip install "data-augment[librosa]"   # optional reference backend
```

## Usage

```python
import numpy as np
from data_augment import Augmenter

aug = Augmenter(sample_rate=16000)       # seed=<int> if you want reproducible noise

# The default chain gives 5 x 2 x 2 = 20 possible augmentations per input (pitch, noise, phase_flip).
variants = aug.augment(x)

# Dataset in, bigger dataset out.
augmented = aug.augment([x1, x2, x3])    # or np.stack([...]) for a 2-D batch
# len(augmented) == len(input) * num_combinations

# Explicit chain — cartesian product across the whole thing.
out = aug.augment(x, [
    ("pitch",      {"semitones": [-2.0, 2.0]}),
    ("gain",       {"db":        [-3.0, 3.0]}),
    ("noise",      {"snr_db":    [20.0], "kind": ["white"]}),
    ("phase_flip", {"flip":      [False, True]}),
])
# len(out) == 2 * 2 * 1 * 2 == 8
```

### Augmentations

- `pitch` — PSOLA shift, length-preserving. `semitones`: float or list.
- `gain` — dB scaling. `db`: float or list.
- `phase_flip` — sign inversion. `flip`: bool or list.
- `noise` — additive colored noise at target SNR. `snr_db`: float or list; `kind`: `"white"` | `"pink"` | `"brown"` or list.
- `speed` — resample; changes pitch too. `factor`: float or list.
- `stretch` — PSOLA time-scale; preserves pitch. `factor`: float or list.
- `apf` — all-pass cascade (phase-only). `sections`: list of `{"order": 1 | 2, "freq_hz": ..., "q": ... (2nd-order only)}`. Not cartesian-expanded — `sections` is one filter; chain more `("apf", {...})` entries if you want additional all-passes.

### Data shape

- Each audio sample is **1-D mono**. Stereo: split by channel first, don't hand it `(2, N)` and hope.
- **Every clip in a batch must be the same length.** Pre-process (random-crop + zero-pad to a fixed N) before calling `augment`. Variable-length lists are rejected.
- For 2-D inputs, shape is `(batch, samples)` — `(B, N)`. **Not** `(N, B)`. The library flags obvious transpositions (trailing dim < 50 samples with a much larger leading dim) and tells you what shape it expected.
- Equivalent forms: `x` (1-D), `np.stack([x1, x2, x3])` (2-D), and `[x1, x2, x3]` (list) all produce the same result when the lengths match.

## API

`Augmenter(sample_rate=16000, oversample_factor=4, sinc_kernel_size=4097, seed=None)` — only `noise` uses the seed; it's re-applied per `augment()` call.

`augment(data, augmentations=None) -> list[AudioSignal]` — flat list of length `num_inputs × num_combinations`, same backend type (numpy/torch) as the input. `augmentations=None` uses `DEFAULT_AUGMENTATIONS`.

## Notes

**Length is preserved.** `speed` and `stretch` change length internally; the output is center-cropped / zero-padded back to each input's length before return. Everything stacks.

**Pitch backend.** Pure NumPy/SciPy port of `librosa.pyin`, bit-equivalent (`atol=1e-12`) across 100 parametrized cases. Install the `librosa` extra if you want it as a fallback; otherwise the native backend is what ships and what runs.

**PSOLA is a hack at best, cheap party trick at worst, you be lying if you don't think so.** I'm not going to pretend otherwise. Time-Domain PSOLA (TD-PSOLA) has been the workhorse of pitch-shifting for three decades because nothing cheaper sounds better, but the algorithm has genuinely embarrassing failure modes and we inherit every one of them. Smadge
- **Pure tones are a trap.** Feed TD-PSOLA a continuous sine at 150 Hz and ask it to shift down an octave. The algorithm "succeeds" by emitting new peaks at 75 Hz spacing, but each grain it emits is still two periods of the *same 150 Hz sine*. The output is 150 Hz, because every grain of the output contains 150 Hz content and the grains are stitched together with COLA. PSOLA is not a frequency shifter; it's a pitch-*pulse*-relocator. If your input doesn't have impulsive structure at the pitch rate, the algorithm has nothing to relocate, and you get back what you put in. This is also why ±12 semitone shifts look fine on real voice and explode on synthetic harmonic stacks.
- **I forgor to overlap-add with triangle windows sized to the new-peak spacing.** I forgot this trick, and tried Hann windows sized to `2·input_period`. If you've read pre-2002 PSOLA papers and you're wondering why their audio samples sound like a dying modem, that's why.
- **Don't shift more than 400 cents (4 semitones) if you like staying data-hygenic** Just don't.

**Edge artifacts.** Reflection-padded sinc resampling leaves <1% of peak in the first/last ~0.1 s at 16 kHz. Below machine-audible range for typical clips; not a length concern.
