import argparse
import logging

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def onnx_quantize(model_dir: str, model_name: str, output_dir: str) -> None:
    operators_to_quantize = [
        "MatMul",
        "Attention",
        "Gather",
        "LSTM",
        "Transpose",
        "EmbedLayerNormalization",
    ]
    dqconfig = AutoQuantizationConfig.avx512_vnni(
        is_static=False, per_channel=False, operators_to_quantize=operators_to_quantize
    )
    quantizer = ORTQuantizer.from_pretrained(model_dir, file_name=model_name)

    quantizer.quantize(
        save_dir=output_dir,
        quantization_config=dqconfig,
    )

    logger.info(f"Quantized model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize an onnx model.")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name .onnx"
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Path to model directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the ONNX model",
    )
    args = parser.parse_args()

    onnx_quantize(
        model_name=args.model_name,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
    )
