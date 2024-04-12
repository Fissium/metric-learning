import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
from oml.const import (
    CATEGORIES_COLUMN,
    IS_GALLERY_COLUMN,
    IS_QUERY_COLUMN,
    LABELS_COLUMN,
    OBLIGATORY_COLUMNS,
    SPLIT_COLUMN,
)

OBLIGATORY_COLUMNS.remove("path")

OBLIGATORY_COLUMNS = [*OBLIGATORY_COLUMNS, "text"]


def check_retrieval_dataframe_format(
    df: Path | str | pd.DataFrame,
    sep: str = ",",
    verbose: bool = True,
) -> None:
    if isinstance(df, Path | str):
        df = pd.read_csv(df, sep=sep, index_col=None)

    assert all(x in df.columns for x in OBLIGATORY_COLUMNS), df.columns

    assert set(df[SPLIT_COLUMN]).issubset({"train", "validation"}), set(
        df[SPLIT_COLUMN]
    )

    mask_train = df[SPLIT_COLUMN] == "train"

    if mask_train.sum() > 0:
        q_train_vals = df[IS_QUERY_COLUMN][mask_train].unique()  # type: ignore
        assert pd.isna(q_train_vals[0]) and len(q_train_vals) == 1, q_train_vals
        g_train_vals = df[IS_GALLERY_COLUMN][mask_train].unique()  # type: ignore
        assert pd.isna(g_train_vals[0]) and len(g_train_vals) == 1, g_train_vals

    val_mask = ~mask_train

    if val_mask.sum() > 0:
        for split_field in [IS_QUERY_COLUMN, IS_GALLERY_COLUMN]:
            unq_values = set(df[split_field][val_mask])
            assert unq_values in [{False}, {True}, {False, True}], unq_values
        assert all(
            (
                (df[IS_QUERY_COLUMN][val_mask].astype(bool))
                | df[IS_GALLERY_COLUMN][val_mask].astype(bool)
            ).to_list()  # type: ignore
        )

    assert df[LABELS_COLUMN].dtypes == int

    # we explicitly put ==True here because of Nones
    labels_query = set(df[LABELS_COLUMN][df[IS_QUERY_COLUMN] == True])  # noqa
    labels_gallery = set(df[LABELS_COLUMN][df[IS_GALLERY_COLUMN] == True])  # noqa
    assert labels_query.intersection(labels_gallery) == labels_query

    if CATEGORIES_COLUMN in df.columns:
        label_to_category = defaultdict(set)
        for _, row in df.iterrows():
            label_to_category[row[LABELS_COLUMN]].add(row[CATEGORIES_COLUMN])

        bad_categories = {k: v for k, v in label_to_category.items() if len(v) > 1}

        if bad_categories and verbose:
            warnings.warn(
                f"Note! You mapping between categories and labels is not bijection!"
                f"During the training and validation we will force it to be bijection"
                f"by picking one random category for each label."
                f"\n"
                f"{bad_categories}",
                stacklevel=2,
            )
