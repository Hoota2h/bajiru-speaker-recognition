# Bajiru Speaker Recognition (Python)

A testing ground for prototypes in Python.

This project uses [`uv`](https://docs.astral.sh/uv/) to manage the Python environments.

Running `uv sync` at the root of the repository will install all dependencies required. Any `uv run` commands here can also be ran from the root of the repository.

## Baji Recognition

### Setup and Run

To run the ML model, you will need a Hugging Face API token, stored in your environment variables under `HUGGING_TOKEN`. Otherwise, a more simple method is the default

- `ffmpeg` may need to be installed manually on your system
- For Linux environments, you may need additional dependencies to install `pyaudio`.

    ```bash
    sudo apt update
    sudo apt install portaudio19-dev python3-dev
    ```

Then you can use the entrypoint `uv run detect_speaker` which will start ingesting microphone audio.

### Scripts

All other scripts can be run via `uv run script_name` and accept a `-h` or `--help` argument

- `quick_vis` - get quick stats & output demo from preset input data, args determine which types of visualization to do
- `vis` - get stats for arbitrary file, can output plots and/or generate demo video

Aforementioned demo video is a small square of color, intended to represent the eye color of the speaker identified. Currently, colors
are aligned with the *start* of the sound chunk that was processed, rather than afterwards (the latter would be more representative of
processing/chunking latency)

## Pitch Math

A simple ML model designed to parse through audio mathematically and then pass the results to a classifier.

A model first needs to be trained based on known audio files:

```bash
uv run pitch-train LOW_DIR HIGH_DIR [--model PATH] [--verbose]
```

The model can then be tested on sample data to verify correctness:

```bash
uv run pitch-test PATH [--model PATH] [--log FILE] [--verbose]
```

Finally, test the model with live audio:

```bash
uv run pitch-live [--model PATH] [--device INDEX] [--verbose]
```

## Development

```bash
uv sync --all-extras

uv run pre-commit install
```

To run the automated tests, use `uv run pytest ./tests`
