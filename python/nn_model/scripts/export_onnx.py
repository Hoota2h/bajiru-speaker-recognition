import argparse
import model_preset
import onnx
import torch


def strip_metadata(path_in, path_out):
    m = onnx.load(path_in)
    if m.metadata_props:
        m.metadata_props.clear()

    g = m.graph
    g.metadata_props.clear()

    for node in g.node:
        node.metadata_props.clear()

    onnx.save(m, path_out)


segment_samples = model_preset.segment_samples
win_length = model_preset.win_length
hop_length = model_preset.hop_length
model = model_preset.model.to("cpu")


def export_checkpoint(checkpoint_path: str, onnx_path: str):
    model.load_state_dict(torch.load(checkpoint_path)["model"], strict=False)

    model.eval()
    dummy = torch.randn(segment_samples)
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        export_params=True,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={
            "x": {
                0: torch.export.Dim.STATIC,
            },
        },
    )
    strip_metadata(onnx_path, onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=False)
    parser.add_argument("output", nargs=1)
    args = parser.parse_args()

    if args.checkpoint:
        export_checkpoint(args.checkpoint, args.output[0])
