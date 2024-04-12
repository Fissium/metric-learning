import os
from collections.abc import Callable
from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from oml.const import (
    EMBEDDINGS_KEY,
    HYDRA_BEHAVIOUR,
    IS_GALLERY_KEY,
    IS_QUERY_KEY,
    LABELS_COLUMN,
    LABELS_KEY,
    SPLIT_COLUMN,
)
from oml.interfaces.models import IExtractor
from oml.lightning.callbacks.metric import (
    MetricValCallback,
    MetricValCallbackDDP,
)
from oml.lightning.pipelines.parser import (
    check_is_config_for_ddp,
    parse_ckpt_callback_from_config,
    parse_engine_params_from_config,
    parse_logger_from_config,
    parse_scheduler_from_config,
)
from oml.metrics.embeddings import EmbeddingMetrics, EmbeddingMetricsDDP
from oml.registry.losses import get_criterion_by_cfg
from oml.registry.optimizers import get_optimizer_by_cfg
from oml.registry.samplers import get_sampler_by_cfg
from oml.utils.misc import dictconfig_to_dict, set_global_seed
from transformers import AutoModel, AutoTokenizer

from src.metric_learning.dataset import DatasetQueryGallery, TextDataset
from src.metric_learning.lightning import (
    BertExtractorModule,
    BertExtractorModuleDDP,
)
from src.metric_learning.model import BertExtractor
from src.metric_learning.utils.dataframe_format import (
    check_retrieval_dataframe_format,
)

torch.set_float32_matmul_precision("high")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_extractor(cfg: DictConfig) -> IExtractor:
    base_model = AutoModel.from_pretrained(cfg.extractor.name)

    extractor = BertExtractor(base_model, **cfg.extractor.args)

    return extractor


def get_tokenizer(cfg: DictConfig) -> Callable:
    return AutoTokenizer.from_pretrained(cfg.tokenizer.name)


def get_dataloaders(
    cfg: DictConfig, tokenizer: Callable
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    df = pd.read_csv(
        Path(cfg.dataset_root, cfg.dataframe_name).resolve(), index_col=None
    )

    check_retrieval_dataframe_format(df, verbose=True)

    mapper = {
        L: i
        for i, L in enumerate(df.sort_values(by=[SPLIT_COLUMN])[LABELS_COLUMN].unique())
    }

    train_df = df.loc[df[SPLIT_COLUMN] == "train"].reset_index(drop=True).copy()
    train_df[LABELS_COLUMN] = train_df[LABELS_COLUMN].map(mapper)

    train_dataset = TextDataset(
        tokenizer=tokenizer, dataframe=train_df, **cfg.tokenizer.args
    )

    val_df = df.loc[df[SPLIT_COLUMN] == "validation"].reset_index(drop=True).copy()
    val_dataset = DatasetQueryGallery(
        tokenizer=tokenizer, dataframe=val_df, **cfg.tokenizer.args
    )
    sampler_runtime_args = {
        "labels": train_dataset.get_labels(),
        "label2category": train_dataset.get_label2category(),
    }
    sampler = (
        get_sampler_by_cfg(cfg.sampler, **sampler_runtime_args)
        if cfg.sampler is not None
        else None
    )

    if sampler is None:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=cfg.num_workers,
            batch_size=cfg.bs_train,
            drop_last=True,
            shuffle=True,
        )

    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=cfg.num_workers,
        )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.bs_val,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    return train_loader, val_loader


@hydra.main(
    config_path="./configs",
    config_name="train_bert.yaml",
    version_base=HYDRA_BEHAVIOUR,
)
def main_hydra(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)

    extractor = get_extractor(cfg)
    tokenizer = get_tokenizer(cfg)
    train_loader, val_loader = get_dataloaders(cfg, tokenizer=tokenizer)

    cfg = dictconfig_to_dict(cfg)  # type: ignore
    criterion = get_criterion_by_cfg(
        cfg["criterion"],
        **{"label2category": train_loader.dataset.get_label2category()},
    )
    optimizable_parameters = [
        {
            "lr": cfg["optimizer"]["args"]["lr"],
            "params": extractor.parameters(),
        },
        {
            "lr": cfg["optimizer"]["args"]["lr"],
            "params": criterion.parameters(),
        },
    ]
    optimizer = get_optimizer_by_cfg(
        cfg["optimizer"],
        **{"params": optimizable_parameters},  # type: ignore
    )

    trainer_engine_params = parse_engine_params_from_config(cfg)
    is_ddp = check_is_config_for_ddp(trainer_engine_params)

    module_kwargs = {}
    module_kwargs.update(parse_scheduler_from_config(cfg, optimizer=optimizer))
    if is_ddp:
        module_kwargs.update({"loaders_train": train_loader, "loaders_val": val_loader})
        module_constructor = BertExtractorModuleDDP

    else:
        module_constructor = BertExtractorModule

    metrics_constructor = EmbeddingMetricsDDP if is_ddp else EmbeddingMetrics

    metrics_calc = metrics_constructor(
        embeddings_key=EMBEDDINGS_KEY,
        labels_key=LABELS_KEY,
        is_query_key=IS_QUERY_KEY,
        is_gallery_key=IS_GALLERY_KEY,
        **cfg.get("metric_args", {}),
    )

    metrics_clb_constructor = MetricValCallbackDDP if is_ddp else MetricValCallback

    metrics_clb = metrics_clb_constructor(
        metric=metrics_calc,  # type: ignore
        log_images=cfg.get("log_images", False),
    )

    logger = parse_logger_from_config(cfg)
    logger.log_pipeline_info(cfg)

    pl_module = module_constructor(
        extractor=extractor,
        criterion=criterion,
        optimizer=optimizer,
        labels_key=train_loader.dataset.labels_key,
        freeze_n_epochs=cfg.get("freeze_n_epochs", 0),
        **module_kwargs,
    )

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg["valid_period"],
        default_root_dir=str(Path.cwd()),
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=[metrics_clb, parse_ckpt_callback_from_config(cfg)],
        logger=logger,
        precision=cfg.get("precision", 32),
        **trainer_engine_params,
        **cfg.get("lightning_trainer_extra_args", {}),
    )
    if is_ddp:
        trainer.fit(model=pl_module)
    else:
        trainer.fit(
            model=pl_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )


if __name__ == "__main__":
    main_hydra()
