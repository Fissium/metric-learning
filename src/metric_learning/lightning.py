from typing import Any

import torch
from oml.lightning.modules.extractor import ExtractorModule, ExtractorModuleDDP


class BertExtractorModule(ExtractorModule):
    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        embeddings = self.model(batch)
        bs = len(embeddings)

        loss = self.criterion(embeddings, batch[self.labels_key])  # type: ignore
        loss_name = (getattr(self.criterion, "criterion_name", "") + "_loss").strip("_")
        self.log(
            loss_name,
            loss.item(),
            prog_bar=True,
            batch_size=bs,
            on_step=True,
            on_epoch=True,
        )

        if hasattr(self.criterion, "last_logs"):
            self.log_dict(
                self.criterion.last_logs,  # type: ignore
                prog_bar=False,
                batch_size=bs,
                on_step=True,
                on_epoch=False,
            )

        if self.scheduler is not None:
            self.log(
                "lr",
                self.scheduler.get_last_lr()[0],
                prog_bar=True,
                batch_size=bs,
                on_step=True,
                on_epoch=False,
            )

        return loss

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
        *_: Any,
    ) -> dict[str, Any]:
        embeddings = self.model.extract(batch)  # type: ignore
        return {**batch, **{self.embeddings_key: embeddings}}


class BertExtractorModuleDDP(ExtractorModuleDDP):
    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        embeddings = self.model(batch)
        bs = len(embeddings)

        loss = self.criterion(embeddings, batch[self.labels_key])  # type: ignore
        loss_name = (getattr(self.criterion, "criterion_name", "") + "_loss").strip("_")
        self.log(
            loss_name,
            loss.item(),
            prog_bar=True,
            batch_size=bs,
            on_step=True,
            on_epoch=True,
        )

        if hasattr(self.criterion, "last_logs"):
            self.log_dict(
                self.criterion.last_logs,  # type: ignore
                prog_bar=False,
                batch_size=bs,
                on_step=True,
                on_epoch=False,
            )

        if self.scheduler is not None:
            self.log(
                "lr",
                self.scheduler.get_last_lr()[0],
                prog_bar=True,
                batch_size=bs,
                on_step=True,
                on_epoch=False,
            )

        return loss

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
        *_: Any,
    ) -> dict[str, Any]:
        embeddings = self.model.extract(batch)  # type: ignore
        return {**batch, **{self.embeddings_key: embeddings}}
