import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from oml.const import (
    CATEGORIES_COLUMN,
    IS_GALLERY_COLUMN,
    IS_QUERY_COLUMN,
    LABELS_COLUMN,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def switch_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "matching":
        df = df.rename(columns={"group_id": "label"})
    elif mode == "search":
        df.loc[df.item_id.isna(), "item_id"] = df.loc[df.item_id.isna()].good_cod
        df["good_cod"] = df.groupby(["item_id"])["good_cod"].transform("min")
        df = df.rename(columns={"good_cod": "label"})

    else:
        logger.error(f"Mode {mode} not in [search, matching]")
        raise NotImplementedError

    return df


def replace_analogs(df: pd.DataFrame, dct_analogs_path: str) -> pd.DataFrame:
    dct_analogs = pd.read_csv(Path(dct_analogs_path))
    df = pd.merge(df, dct_analogs, how="left", on=["good_cod"])
    df.loc[~df.good_analog_cod.isna(), "good_cod"] = df.loc[
        ~df.good_analog_cod.isna()
    ].good_analog_cod
    df = df.drop(columns=["good_analog_cod"])
    return df


def replace_duplicates(df: pd.DataFrame, dct_duplicates_path: str) -> pd.DataFrame:
    dct_duplicates = pd.read_csv(Path(dct_duplicates_path))
    df = pd.merge(df, dct_duplicates, how="left", on=["good_cod"])
    df.loc[~df.good_analog_cod.isna(), "good_cod"] = df.loc[
        ~df.good_analog_cod.isna()
    ].good_analog_cod
    df = df.drop(columns=["good_analog_cod"])
    return df


def convert_vit_df(
    dataframe_path: str,
    output_dir: str,
    dct_analogs_path: str | None,
    dct_duplicates_path: str | None,
    mode: str,
) -> None:
    df = pd.read_csv(Path(dataframe_path).resolve())

    if dct_analogs_path is not None:
        df = replace_analogs(df, dct_analogs_path)

    if dct_duplicates_path is not None:
        df = replace_duplicates(df, dct_duplicates_path)

    df = switch_mode(df, mode=mode)

    df["labels_per_category"] = df.groupby(CATEGORIES_COLUMN)[LABELS_COLUMN].transform(
        "nunique"
    )

    df = df.loc[df["labels_per_category"] > 1]

    df["categories_per_label"] = df.groupby(LABELS_COLUMN)[CATEGORIES_COLUMN].transform(
        "nunique"
    )
    df = df.loc[df["categories_per_label"] == 1]

    df["n_instances"] = df.groupby([LABELS_COLUMN]).transform("size")
    df = df.loc[df["n_instances"] >= 2].reset_index(drop=True)

    df["path"] = (
        df["path"].str.split(Path(output_dir).resolve().as_posix()).str[-1].str[1:]
    )

    classes = df[LABELS_COLUMN].unique()
    idxs = df.loc[df.label.isin(classes)].drop_duplicates(subset=[LABELS_COLUMN]).index
    category = np.array(df.loc[idxs][CATEGORIES_COLUMN])

    train_classes, validation_classes = train_test_split(
        classes, test_size=0.1, random_state=42, stratify=category
    )

    train_indicies = df.loc[df[LABELS_COLUMN].isin(train_classes)].index.to_numpy()
    validation_indicies = df.loc[
        df[LABELS_COLUMN].isin(validation_classes)
    ].index.to_numpy()

    logger.info(f"Median size of n_instance: {df.n_instances.median()}")

    train_df = df.loc[train_indicies].reset_index(drop=True)
    train_df["split"] = "train"
    val_df = df.loc[validation_indicies].reset_index(drop=True)
    val_df["split"] = "validation"

    val_df[IS_QUERY_COLUMN] = True
    val_df[IS_GALLERY_COLUMN] = True

    # Saving the dataframe to output directory
    logger.info(f"Dataframe train_vit_df.csv saved to {output_dir}")
    pd.concat([train_df, val_df]).drop(
        columns=["n_instances", "categories_per_label", "labels_per_category"]
    ).reset_index(drop=True).to_csv(
        Path(output_dir).joinpath("train_vit_df.csv"), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert text dataframe.")
    parser.add_argument(
        "--dataframe_path", type=str, required=True, help="Path to the input dataframe"
    )
    parser.add_argument(
        "--dct_analogs",
        type=str,
        default=None,
        help="Path to the ds_dct_goods_analogs df (good_cod, good_analog_cod)",
    )
    parser.add_argument(
        "--dct_duplicates",
        type=str,
        default=None,
        help="Path to the ds_dct_goods_duplicates df (good_cod, good_analog_cod)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output dataframes",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="search",
        help="Mode: matching (good_cod=group_id) or search",
    )
    args = parser.parse_args()

    convert_vit_df(
        dataframe_path=args.dataframe_path,
        output_dir=args.output_dir,
        dct_analogs_path=args.dct_analogs,
        dct_duplicates_path=args.dct_duplicates,
        mode=args.mode,
    )
