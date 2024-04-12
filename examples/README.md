# Examples

## Remove image duplicates

Some products contain identical images.

To detect and remove all such images, the `imagededup` library is used:

```bash
python remove_duplicates.py --input_dir=/path/to/images --dry_run
```

If the --dry_run flag is specified, the images will not be deleted, and the result will be saved in the duplicates.json file.

## Build ViT Dataset

The dataset for training the ViT model is created based on the `dud.xml` feed.

**Important**: `extra_dir` should be placed next to the `output_dir` directory.

You can also specify a directory with additional images.

```
usage: build_vit_dataset.py [-h] --xml_url XML_URL --output_dir OUTPUT_DIR [--extra_dir EXTRA_DIR] [--num_workers NUM_WORKERS]

Build vit dataset.

options:
  -h, --help            show this help message and exit
  --xml_url XML_URL     Path to the xml file to parse
  --output_dir OUTPUT_DIR
                        Directory to save images
  --extra_dir EXTRA_DIR
                        Extra directory with additional images
  --num_workers NUM_WORKERS
                        Number of workers to download images
```

## Convert dataset to OML format

When converting the dataset for the `vit` model, you can additionally specify a dataframe with references to analogs and/or duplicates.

Also, the `output_dir` should be placed next to the image directories.

```
usage: vit_converter.py [-h] --dataframe_path DATAFRAME_PATH [--dct_analogs DCT_ANALOGS] [--dct_duplicates DCT_DUPLICATES] --output_dir OUTPUT_DIR [--mode MODE]

Convert text dataframe.

options:
  -h, --help            show this help message and exit
  --dataframe_path DATAFRAME_PATH
                        Path to the input dataframe
  --dct_analogs DCT_ANALOGS
                        Path to the ds_dct_goods_analogs df (good_cod, good_analog_cod)
  --dct_duplicates DCT_DUPLICATES
                        Path to the ds_dct_goods_duplicates df (good_cod, good_analog_cod)
  --output_dir OUTPUT_DIR
                        Directory to save the output dataframes
  --mode MODE           Mode: matching (good_cod=group_id) or search
```

For converting the dataset for the `bert` model:

```
usage: bert_converter.py [-h] --dataframe_path DATAFRAME_PATH --output_dir OUTPUT_DIR

Convert text dataframe.

options:
  -h, --help            show this help message and exit
  --dataframe_path DATAFRAME_PATH
                        Path to the input dataframe
  --output_dir OUTPUT_DIR
                        Directory to save the output dataframes
```

## Export to ONNX

The `optimum` library is used to export models to the `onnx` format:

For `bert`

```
usage: bert_to_onnx.py [-h] --model_name MODEL_NAME [--model_weights MODEL_WEIGHTS] --output_dir OUTPUT_DIR [--normalize_features]

Export model to ONNX format.

options:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Model name (huggingface hub)
  --model_weights MODEL_WEIGHTS
                        Path to model weights
  --output_dir OUTPUT_DIR
                        Output directory for the ONNX model
  --normalize_features  Normalize features or not
```

For `vit`

```
usage: vit_to_onnx.py [-h] --model_arch MODEL_ARCH [--model_weights MODEL_WEIGHTS] --output_dir OUTPUT_DIR [--normalize_features]

Export model to ONNX format.

options:
  -h, --help            show this help message and exit
  --model_arch MODEL_ARCH
                        Model architecture
  --model_weights MODEL_WEIGHTS
                        Path to model weights
  --output_dir OUTPUT_DIR
                        Output directory for the ONNX model
  --normalize_features  Normalize features or not
```

## Quantization

The `optimum` library is used for model weights quantization:

```
usage: quantize.py [-h] --model_name MODEL_NAME --model_dir MODEL_DIR --output_dir OUTPUT_DIR

Quantize an onnx model.

options:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Model name .onnx
  --model_dir MODEL_DIR
                        Path to model directory
  --output_dir OUTPUT_DIR
                        Output directory for the ONNX model
```
