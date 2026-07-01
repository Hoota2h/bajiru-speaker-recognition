import torch, onnx
import model_preset

checkpoint_path = "checkpoints/last_N.pt"
onnx_path = "model.onnx"


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
