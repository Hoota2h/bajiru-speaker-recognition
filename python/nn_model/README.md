# Audio classifier
Audio classifier using conv-transformer based model.

Accepts an audio buffer and produces a list of scores per each label.

## Preparing the dataset
1) Prepare the audio file and the labels file.
- The audio should have 48000 sample rate, and the format should be 16-bit integer. The file format is `.wav`
- The labels should be exported from audacity to `.txt` file, according to the format described [in the script](scripts/makedata.py#L130).
2) Add all audio/label paths to [the script](scripts/makedata.py#L223).
3) Run the command and the audio/score datasets will be generated.
```py
uv run -s ./python/nn_model/scripts/makedata.py
```

*Note:* The `map_labels` function can be changed to use any other labels format.

## Training the model
1) Configure the training/validation datasets [in the script](scripts/train.py#L55).
2) Run the command and wait until the loss will be less than `0.01`. The training parameters can be tweaked for better results.
```py
uv run -s ./python/nn_model/scripts/train.py
```
To continue the training after tweaks/pause run
```py
uv run -s ./python/nn_model/scripts/train.py -r
```

## Testing the model
Run the command with the checkpoint and input files specified, the output files correspond to the labels.
```py
uv run -s ./python/nn_model/scripts/eval.py -c checkpoints/best_***.pt -i input_audio.wav out_score0.wav out_score1.wav ...
```

## Exporting the model
Run the command with the checkpoint specified, the output will be a onnx model.
```py
uv run -s ./python/nn_model/scripts/export_onnx.py -c checkpoints/best_***.pt model.onnx
```
