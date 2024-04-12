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

Код для обучения моделей с использованием подхода `metric learning`.
В основе проекта лежит библиотека [open-metric-learning](https://github.com/OML-Team/open-metric-learning).

## Requirements

- `python` (протестировано для версии Python `>=3.10, <3.11`).

## Installation

Создать виртуальное окружение, например, с помощь `conda`. Затем установить зависимости:

```bash
pip install -r requirements.txt
```

Дополнительные зависимости (логирование, парсинг фида, удаление дубликатов, экспорт в `onnx`) находятся в файле `requirements_optional.txt`.

Если нужна какая-либо из вышеуказанных возможностей (смотри [`examples`](./examples/)) нужно выполнить:

```bash
pip install -r requirements_optional.txt
```

## Usage

Перед использованием следует ознакомиться с библиотекой [open-metric-learning](https://github.com/OML-Team/open-metric-learning).

Особое внимание нужно уделить [конфигурационному файлу](https://github.com/OML-Team/open-metric-learning/blob/main/pipelines/features_extraction/extractor_cars/train_cars.yaml).

### Dataset

Датасет должен иметь особый [формат](https://open-metric-learning.readthedocs.io/en/latest/oml/data.html).

При обучении текстовой модели, колонка `path` должна быть заменена на `text` и содержать текстовое описание объекта.

Также можно использовать примеры подготовки данных в каталоге [`examples`](./examples):

- Для текстовых моделей - [`examples/bert_converter.py`](./examples/bert_converter.py).
- Для визуальных моделей - [`examples/vit_converter.py`](./examples/vit_converter.py).

## Train

Внеся соответствующие изменения в конфигурационные файлы в директории [`config`](./configs), выполнить:

```bash
python train_bert.py
# OR
python train_vit.py
```

## Examples

Для деплоя моделей может потребоваться их оптимизация.

Для этого можно использовать следующие примеры:

1. Конвертация в `onnx` формат - [`examples/vit_to_onnx.py`](./examples/vit_to_onnx.py) или [`examples/bert_to_onnx.py`](./examples/bert_to_onnx.py).
1. Квантизация - [`examples/quantize.py`](./examples/quantize.py).

## Logging

Для регистрации модели в `mlflow` можно использовать следующий пример:

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

В данном примере регистрируется модель формата `onnx`.
