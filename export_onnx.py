import argparse
from pathlib import Path

import onnx
import torch

from utils.runtime import build_model, normalize_state_dict


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)[-1]
        return out["hm"], out["vaf"], out["haf"]


def remove_identity_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    identity_map = {}
    kept_nodes = []

    for node in graph.node:
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1:
            identity_map[node.output[0]] = node.input[0]
        else:
            kept_nodes.append(node)

    def resolve(name: str) -> str:
        seen = set()
        cur = name
        while cur in identity_map and cur not in seen:
            seen.add(cur)
            cur = identity_map[cur]
        return cur

    for node in kept_nodes:
        for i, name in enumerate(node.input):
            node.input[i] = resolve(name)

    for out in graph.output:
        out.name = resolve(out.name)

    del graph.node[:]
    graph.node.extend(kept_nodes)
    return model


def main():
    parser = argparse.ArgumentParser("Export LaneAF checkpoint to ONNX with fixed batch=1")
    parser.add_argument("--snapshot", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", required=True, help="Path to output .onnx file")
    parser.add_argument("--backbone", default="dla34", choices=["dla34", "erfnet", "enet"])
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--clean-identity", action="store_true", help="Remove Identity nodes after export")
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    output_path = Path(args.output)

    model = build_model(args.backbone, {"hm": 1, "vaf": 2, "haf": 1}, torch.device("cpu"))
    state_dict = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    state_dict = normalize_state_dict(model, state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    wrapped = ExportWrapper(model)
    dummy = torch.randn(1, 3, args.height, args.width)

    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            output_path.as_posix(),
            input_names=["input"],
            output_names=["hm", "vaf", "haf"],
            opset_version=args.opset,
            do_constant_folding=True,
        )

    if args.clean_identity:
        model_proto = onnx.load(output_path.as_posix())
        model_proto = remove_identity_nodes(model_proto)
        onnx.checker.check_model(model_proto)
        onnx.save(model_proto, output_path.as_posix())

    print(output_path)


if __name__ == "__main__":
    main()
