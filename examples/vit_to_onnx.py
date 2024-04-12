import argparse
import logging
from pathlib import Path

import onnx
import onnx.checker
import torch
from oml.models import ViTExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_to_onnx(
    model_arch: str,
    model_weights: str | None,
    output_dir: str,
    normalize_features: bool,
) -> None:
    model = ViTExtractor(
        weights=model_weights, arch=model_arch, normalise_features=normalize_features
    )
    batch_size = 1

    onnx_path = str(Path(output_dir).joinpath(f"{model_arch}.onnx"))

    dummpy_input = torch.randn(batch_size, 3, 224, 224).to("cpu")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummpy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    logger.info(f"ONNX model saved to {onnx_path}")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX format.")
    parser.add_argument(
        "--model_arch", type=str, required=True, help="Model architecture"
    )
    parser.add_argument(
        "--model_weights", type=str, default=None, help="Path to model weights"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the ONNX model",
    )
    parser.add_argument(
        "--normalize_features", action="store_true", help="Normalize features or not"
    )
    args = parser.parse_args()

    export_to_onnx(
        model_arch=args.model_arch,
        model_weights=args.model_weights,
        output_dir=args.output_dir,
        normalize_features=args.normalize_features,
    )
