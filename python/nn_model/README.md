# Audio classifier
Audio classifier using conv-transformer based model.

Accepts an audio buffer and produces a list of scores per each label.

[Training example](scripts/train.py) (the training/validation datasets should be configured)
```py
uv run -s ./python/nn_model/scripts/train.py
```

[Evaluation example](scripts/eval.py)
```py
uv run -s ./python/nn_model/scripts/eval.py -c checkpoints/best_***.pt -i input_audio.wav out_score0.wav out_score1.wav ...
```

[Dataset creation example](scripts/makedata.py)
```py
uv run -s ./python/nn_model/scripts/makedata.py
```

[Onnx export example](scripts/export_onnx.py)
```py
uv run -s ./python/nn_model/scripts/export_onnx.py
```

The [train dataset](src/nn_model/dataset.py) accepts a list of raw audio and target scores files.
The files contain raw binary numbers, the number format should be specified for each file.
- audio file - has additional multiplier property, that can be used to normalize the audio to range `[-1.0:1.0]`. For example for int16 format the audio multiplier should be `1.0 / 0x7FFF`.
- scores file - contains a 2d array of target scores, the scores should be in range [0-1] (that's why there's no multiplier property).
The dataset converts numbers to float32 automatically.