from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
from oml.const import (
    CATEGORIES_COLUMN,
    INPUT_TENSORS_KEY,
    IS_GALLERY_COLUMN,
    IS_GALLERY_KEY,
    IS_QUERY_COLUMN,
    IS_QUERY_KEY,
    LABELS_COLUMN,
    LABELS_KEY,
)
from torch.utils.data import Dataset

from .const import TEXTS_COLUMN


class TextDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: Callable,
        padding: bool,
        truncation: bool,
        max_length: int | None = None,
    ):
        self.labels_key = LABELS_KEY
        self.input_tensors_key = INPUT_TENSORS_KEY
        self.dataframe = dataframe
        self.labels = dataframe[LABELS_COLUMN].values
        texts = list(dataframe[TEXTS_COLUMN].apply(lambda o: str(o)).values)
        self.encodings = tokenizer(
            texts, padding=padding, truncation=truncation, max_length=max_length
        )

    def __getitem__(self, idx) -> dict[str, Any]:
        item = {
            key: torch.tensor(values[idx]) for key, values in self.encodings.items()
        }
        item[LABELS_KEY] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        return len(self.dataframe)

    def get_labels(self) -> np.ndarray:
        return np.array(self.dataframe[LABELS_COLUMN].tolist())

    def get_label2category(self) -> None | dict[int, str | int]:
        if CATEGORIES_COLUMN in self.dataframe.columns:
            label2category = dict(
                zip(
                    self.dataframe[LABELS_COLUMN],
                    self.dataframe[CATEGORIES_COLUMN],
                    strict=True,
                )
            )
        else:
            label2category = None

        return label2category


class DatasetQueryGallery(TextDataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: Callable,
        padding: bool,
        truncation: bool,
        max_length: int | None = None,
        is_query_key: str = IS_QUERY_KEY,
        is_gallery_key: str = IS_GALLERY_KEY,
    ):
        super().__init__(dataframe, tokenizer, padding, truncation, max_length)
        assert all(x in dataframe.columns for x in (IS_QUERY_COLUMN, IS_GALLERY_COLUMN))

        self.is_query_key = is_query_key
        self.is_gallery_key = is_gallery_key

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        item[self.is_query_key] = bool(self.dataframe.iloc[idx][IS_QUERY_COLUMN])
        item[self.is_gallery_key] = bool(self.dataframe.iloc[idx][IS_GALLERY_COLUMN])
        return item
