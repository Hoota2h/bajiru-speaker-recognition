from nn_model.model import AudioClassifier

sample_rate = 44100
segment_samples = 16384
win_length = 1024
hop_length = 512
n_classes = 4
model = AudioClassifier(
    sample_rate=sample_rate,
    n_fft=win_length,
    n_mels=160,
    hop_length=hop_length,
    n_classes=n_classes,
    n_layers=2,
    conv_channels=96,
    d_model=160,
    n_history=1 + segment_samples // hop_length,
)
