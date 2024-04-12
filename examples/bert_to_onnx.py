import argparse
import logging
import sys
from pathlib import Path

import onnx
import onnx.checker
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.append("..")
sys.path.append("../src/metric_learning/")

from src.metric_learning.model import BertExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_to_onnx(
    model_name: str,
    model_weights: str | None,
    output_dir: str,
    normalize_features: bool = True,
) -> None:
    base_model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = BertExtractor(
        base_model=base_model,
        normalise_features=normalize_features,
        weights=model_weights,
    )

    onnx_path = str(Path(output_dir).joinpath(f"{model_name.replace('/', '_')}.onnx"))

    dummy_input = tokenizer(
        "hello, world", padding=True, truncation=True, return_tensors="pt"
    ).to("cpu")

    with torch.no_grad():
        torch.onnx.export(
            model,
            {
                "batch": {
                    "input_ids": dummy_input["input_ids"],
                    "attention_mask": dummy_input["attention_mask"],
                }
            },  # type: ignore
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_len"},
                "attention_mask": {0: "batch_size", 1: "sequence_len"},
                "output": {0: "batch_size"},
            },
        )

    logger.info(f"ONNX model saved to {onnx_path}")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX format.")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name (huggingface hub)"
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
        model_name=args.model_name,
        model_weights=args.model_weights,
        output_dir=args.output_dir,
        normalize_features=args.normalize_features,
    )
