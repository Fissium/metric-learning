import argparse
import logging
from pathlib import Path

import pandas as pd
from oml.const import IS_GALLERY_COLUMN, IS_QUERY_COLUMN
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_text_df(dataframe_path: str, output_dir: str) -> None:
    df = pd.read_csv(Path(dataframe_path).resolve())

    df["num_samples"] = df.groupby(["label"]).transform("size")
    df = df.loc[df.num_samples >= 2].reset_index(drop=True)

    classes = df.label.unique()

    train_classes, validation_classes = train_test_split(
        classes, test_size=0.2, random_state=42
    )

    train_indicies = df.loc[df.label.isin(train_classes)].index.to_numpy()
    validation_indicies = df.loc[df.label.isin(validation_classes)].index.to_numpy()

    train_df = df.loc[train_indicies].reset_index(drop=True)
    train_df["split"] = "train"
    val_df = df.loc[validation_indicies].reset_index(drop=True)
    val_df["split"] = "validation"

    val_df[IS_QUERY_COLUMN] = True
    val_df[IS_GALLERY_COLUMN] = True

    # Saving the dataframe to output directory
    logger.info(f"Dataframe train_bert_df.csv saved to {output_dir}")
    pd.concat([train_df, val_df]).drop(columns=["num_samples"]).reset_index(
        drop=True
    ).to_csv(Path(output_dir).joinpath("train_bert_df.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert text dataframe.")
    parser.add_argument(
        "--dataframe_path", type=str, required=True, help="Path to the input dataframe"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output dataframes",
    )
    args = parser.parse_args()

    convert_text_df(dataframe_path=args.dataframe_path, output_dir=args.output_dir)
