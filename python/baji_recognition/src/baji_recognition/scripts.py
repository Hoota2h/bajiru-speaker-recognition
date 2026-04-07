import argparse
import pathlib
import sys

import numpy as np
from matplotlib import pyplot as plt

from baji_recognition import identifier, process

logger = identifier.logger


def quick_vis(sys_args: list[str] | None = None) -> None:
    if sys_args is None:
        sys_args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Quick visualize performance of defaults",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    optional_args = parser.add_argument_group(description="Optional Arguments:")
    optional_args.add_argument(
        "--video",
        dest="make_video",
        action="store_true",
        help="Generate quick demo video from low.mp3",
    )
    optional_args.add_argument(
        "--series",
        dest="make_series_plots",
        action="store_true",
        help="Make series plots for low and normal voice",
    )
    optional_args.add_argument(
        "--avg",
        dest="make_avgs_plots",
        action="store_true",
        help="Make plots of average values in the low and high bins",
    )
    optional_args.add_argument(
        "--dec",
        dest="make_decisions_plots",
        action="store_true",
        help="Make plots of decision series",
    )
    args = parser.parse_args(sys_args)

    visualize_all(
        make_video=args.make_video,
        make_series_plots=args.make_series_plots,
        make_avgs_plots=args.make_avgs_plots,
        make_decisions_plots=args.make_decisions_plots,
    )
    if not np.any(np.array(list(vars(args).values()))):
        logger.warning("No output args selected")


def visualize(sys_args: list[str] | None = None) -> None:
    if sys_args is None:
        sys_args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Quick visualize performance of defaults",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    optional_args = parser.add_argument_group(description="Optional Arguments:")
    optional_args.add_argument(dest="input_file", type=str, help="Input file")
    optional_args.add_argument(
        dest="output_file", nargs="?", type=str, help="Output file (only required if --vid is set)"
    )
    optional_args.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=identifier.CHUNK_DURATION,
        help="Make plots to visualize data file",
    )
    optional_args.add_argument(
        "--chunk_len",
        dest="chunk_duration",
        action="store_true",
        default=identifier.CHUNK_DURATION,
        help="Duration of chunks to split input file into for processing",
    )
    optional_args.add_argument(
        "--vid",
        dest="make_video",
        action="store_true",
        help="Use flag to generate an output video (requires output_file to be specified)",
    )
    args = parser.parse_args(sys_args)
    if args.plot:
        visualize_file(args.input_file, args.chunk_duration)
    if args.make_video:
        if not args.output_file:
            msg = "Missing output path for video"
            raise ValueError(msg)
        visualize_file(args.input_file, args.chunk_duration)


def visualize_all(
    *, make_video: bool, make_series_plots: bool, make_avgs_plots: bool, make_decisions_plots: bool
) -> None:
    data_dir = pathlib.Path(__file__).parent.parent.parent / "tests" / "data"
    low_voice = data_dir / "low.mp3"
    normal_voice = data_dir / "normal.mp3"
    mixed_sample = data_dir / "sample.mp3"
    chunk_length = identifier.CHUNK_DURATION
    low_video = data_dir / "low.mp4"

    if make_video:
        process.generate_video(low_voice, low_video, chunk_length)
    if make_series_plots or make_avgs_plots or make_decisions_plots:
        low_data = np.array(process.get_values(low_voice, chunk_length))
        normal_data = np.array(process.get_values(normal_voice, chunk_length))
        mixed_data = np.array(process.get_values(mixed_sample, chunk_length))

    if make_series_plots:
        process.plot_series(low_data[:, 0], "low", low_data[:, 1], "high", "Low Data Series")
        process.plot_series(normal_data[:, 0], "low", normal_data[:, 1], "high", "Normal Data Series")
        process.plot_series(mixed_data[:, 0], "low", mixed_data[:, 1], "high", "Mixed Data Series")

    if make_avgs_plots:
        process.plot_series(low_data[:, 0], "low", normal_data[:, 0], "normal", "Low avgs")
        process.plot_series(low_data[:, 1], "low", normal_data[:, 1], "normal", "High avgs")

    if make_decisions_plots:
        process.plot_decisions(low_voice, "Raw Decision", "Low voice decisions", chunk_length)
        process.plot_decisions(normal_voice, "Raw Decision", "Normal voice decisions", chunk_length)
        process.plot_decisions(mixed_sample, "Raw Decision", "Mixed Sample voice decisions", chunk_length)
    if make_series_plots or make_avgs_plots or make_decisions_plots:
        plt.show()


def visualize_file(file: pathlib.Path, chunk_length: float) -> None:

    data = np.array(process.get_values(file, chunk_length))
    process.plot_series(data[:, 0], "low", data[:, 1], "high", "Data Series")
    process.plot_decisions(file, "Raw Decision", "Decisions", chunk_length)
    plt.show()


def make_video(in_file: pathlib.Path, out_file: pathlib.Path, chunk_length: float) -> None:
    process.generate_video(in_file, out_file, chunk_length)
