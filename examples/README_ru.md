# Examples

## Remove image duplicates

Некоторые товары содержат полностью одинаковые изображения.

Чтобы обнаружить все такие изображения и удалить их, используется библиотека `imagededup`:

```bash
python remove_duplicates.py --input_dir=/path/to/images --dry_run
```

Если указан флаг `--dry_run` изображения не будут удалены, а результат будет сохранен в файл `duplicates.json`.

## Build ViT Dataset

Датасет для обучения модели `ViT` создается на основе фида `dud.xml`.

**Важно**. `extra_dir` необходимо поместить рядом с каталогом `output_dir`.

Также можно указать каталог с дополнительными изображениями.

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

При конвертации датасета для модели `vit` дополнительно можно указать датафрейм с справочником аналогов и/или дубликатов.

Также `output_dir` нужно поместить рядом с каталогами изображений.

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

Для конвертаии датасета для модели `bert`:

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

Для экспорта моделей в формат `onnx` используется библиотека `optimum`:

Для `bert`

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

Для `vit`

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

Для квантизации весов моделей используется библиотека `optimum`:

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
