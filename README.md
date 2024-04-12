<!--ts-->

- [About](#about)
    - [Requirements](#requirements)
    - [Installation](#installation)
    - [Usage](#usage)
        - [Dataset](#dataset)
    - [Train](#train)
    - [Examples](#examples)
    - [Logging](#logging)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

<!-- Added by: fissium, at: Fri Apr  5 02:25:57 PM MSK 2024 -->

<!--te-->

# About

Code for training models using the `metric learning` approach.
The project is based on the library [open-metric-learning](https://github.com/OML-Team/open-metric-learning).

## Requirements

- `python` (tested with Python version `>=3.10, <3.11`).

## Installation

Create a virtual environment, for example, using conda. Then install the dependencies:

```bash
pip install -r requirements.txt
```

Additional dependencies (logging, feed parsing, duplicate removal, export to onnx) are listed in the file `requirements_optional.txt`.

If any of the above-mentioned features are needed (refer to [examples](./examples/)), execute:

```bash
pip install -r requirements_optional.txt
```

## Usage

Before usage, familiarize yourself with the library [open-metric-learning](https://github.com/OML-Team/open-metric-learning).

Particular attention should be given to the [configuration file](https://github.com/OML-Team/open-metric-learning/blob/main/pipelines/features_extraction/extractor_cars/train_cars.yaml).

### Dataset

The dataset should have a specific [format](https://open-metric-learning.readthedocs.io/en/latest/oml/data.html).

When training a text model, the column `path` should be replaced with `text` and contain the textual description of the object.

Examples of data preparation are available in the examples directory:

- For text models - [examples/bert_converter.py](./examples/bert_converter.py).
- For visual models - [examples/vit_converter.py](./examples/vit_converter.py).

## Train

After making the necessary changes to the configuration files in the [configs](./configs) directory, execute:

```bash
python train_bert.py
# OR
python train_vit.py
```

## Examples

Model optimization may be required for deployment.

The following examples can be used:

- Conversion to onnx format - [examples/vit_to_onnx.py](./examples/vit_to_onnx.py) or [examples/bert_to_onnx.py](./examples/bert_to_onnx.py).
- Quantization - [examples/quantize.py](./examples/quantize.py).

## Logging

To register a model in `mlflow`, you can use the following example:

```python

import mlflow
import onnx

mlflow.set_tracking_uri("http://localhost:8000")

model_path = "./ViTExtractor.onnx"
artifact_path = "./artifacts"
onnx_model = onnx.load(model_path)

with mlflow.start_run(experiment_id="1") as run:
    mlflow.onnx.log_model(onnx_model, "model", save_as_external_data=False)
    mlflow.log_artifact(artifact_path)
```

In this example, a model of onnx format is registered.
