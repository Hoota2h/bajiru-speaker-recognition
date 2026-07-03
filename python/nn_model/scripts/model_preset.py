from nn_model.model import AudioClassifier

sample_rate = 48000
segment_samples = 24000
win_length = 2400
hop_length = 800
n_classes = 5
model = AudioClassifier(
    sample_rate=sample_rate,
    n_fft=win_length,
    hop_length=hop_length,
    n_classes=n_classes,
    n_heads=4,
    n_attns=3,
    d_model=128,
    n_mels=320,
    n_channels=128,
    n_convs=3,
    n_history=1 + (segment_samples - win_length) // hop_length,
)
