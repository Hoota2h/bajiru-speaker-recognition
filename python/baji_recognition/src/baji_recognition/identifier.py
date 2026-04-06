import time
from collections.abc import Buffer, Mapping

import librosa
import numpy as np
import pyaudio
import torch
from dotenv import dotenv_values

# ml testing
from pyannote.audio import Pipeline
from pyaudio import PyAudio, Stream

from baji_recognition import logging

logging.set_up_logging()
logger = logging.logger

N_FFT = 512
HOP = (int)(512 / 4)
RATE = 44100
CHUNK_DURATION = 1
VOICE_THRESH = 380
SILENCE_CUTOFF = -45


class Thing:
    def __init__(self) -> None:
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = RATE
        # Allow easier configuration of chunk size by tying it to the sample rate and chunk duration in seconds
        self.chunk = int(RATE * CHUNK_DURATION)
        self.p: PyAudio
        self.stream: Stream
        # ml testing
        environment = dotenv_values(".env")
        if "HUGGING_TOKEN" in environment:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=environment["HUGGING_TOKEN"],
            )
        else:
            self.pipeline = None
            logger.warning(
                "No HUGGING_TOKEN Face token found to retrieve pretrained model, defaulting to simple threshold"
            )
        # self.pipeline.to(torch.device("cuda"))

    def start(self) -> None:
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            output=False,
            stream_callback=self.callback,  # type: ignore [arg-type]
            frames_per_buffer=self.chunk,
        )

    def stop(self) -> None:
        self.stream.close()
        self.p.terminate()

    def callback(
        self, in_data: Buffer, frame_count: int, time_info: Mapping[str, float], flag: int
    ) -> tuple[bytes | None, int]:
        logger.debug("Frame_count: %d", frame_count)
        logger.debug("Time info: %s", time_info)
        logger.debug("Flag: %s", flag)
        data = np.frombuffer(in_data, dtype=np.float32)
        logger.info("Processing chunk")
        if self.pipeline is not None:
            # ml testing
            test = torch.tensor(data).reshape((2, -1))
            output = self.pipeline({"waveform": test, "sample_rate": self.rate})  # runs locally
        else:
            output = process(data, self.rate)[0]

        logger.info(output)

        return None, pyaudio.paContinue

    def waiter(self) -> None:
        while self.stream.is_active():
            time.sleep(2.0)


def process(data: np.ndarray, rate: int) -> tuple[str, float, float]:
    transformed = librosa.stft(data, n_fft=N_FFT)
    freqs = librosa.fft_frequencies(sr=rate, n_fft=N_FFT)
    return identify_voice_type(transformed, freqs, VOICE_THRESH)


def identify_voice_type(transformed: np.ndarray, freqs: np.ndarray, threshold: int) -> tuple[str, float, float]:
    split_bin = np.argmax(freqs >= threshold)

    lowband = transformed[:split_bin, :]
    highband = transformed[split_bin:, :]

    y_low = librosa.istft(lowband, hop_length=HOP)
    y_high = librosa.istft(highband, hop_length=HOP)
    peak_amplitude = np.max(np.abs(transformed))  # Peak volume in amplitude
    peak_db = librosa.amplitude_to_db(peak_amplitude, ref=1.0)

    low = np.average(y_low)
    high = np.average(y_high)
    logger.debug("Low avg: %f", low)
    logger.debug("High avg: %f", high)

    logger.debug("Peak amplitude: %f", peak_amplitude)
    logger.debug("Peak dB: %f", peak_db)

    if peak_db < SILENCE_CUTOFF:
        logger.debug("below min db skipping")
        decision = "unknown"
    else:
        decision = ["high", "low"][bool(low > high)]
    return decision, low, high


def main() -> None:
    audio = Thing()
    audio.start()  # open the mic stream
    audio.waiter()
    audio.stop()


if __name__ == "__main__":
    main()
