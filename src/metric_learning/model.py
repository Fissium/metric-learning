from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from oml.interfaces.models import IExtractor
from oml.models.utils import (
    remove_criterion_in_state_dict,
)
from oml.utils.misc_torch import normalise


def remove_prefix_from_state_dict(
    state_dict: OrderedDict[str, torch.Tensor], trial_key: str
) -> OrderedDict[str, torch.Tensor]:
    if trial_key == "":
        return state_dict

    else:
        for k in list(state_dict.keys()):
            if k.startswith(trial_key):
                state_dict[k[len(trial_key) :]] = state_dict[k]
                del state_dict[k]

        print(f"Prefix <{trial_key}> was removed from the state dict.")

        return state_dict


class BertExtractor(IExtractor):
    def __init__(
        self,
        base_model: nn.Module,
        weights: None | Path | str = None,
        normalise_features=False,
    ):
        super().__init__()
        self.model = base_model
        self.normalise_features = normalise_features

        if weights is None:
            return

        ckpt = torch.load(weights, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        state_dict = remove_criterion_in_state_dict(state_dict)
        ckpt = remove_prefix_from_state_dict(state_dict, trial_key="model.model.")
        self.model.load_state_dict(ckpt, strict=True)

    def get_features(self, batch: dict[str, Any]):
        output = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        last_hidden_state = (
            output.last_hidden_state
        )  # shape: (batch_size, seq_length, bert_hidden_dim)
        CLS_token_state = last_hidden_state[
            :, 0, :
        ]  # obtaining CLS token state which is the first token.
        return CLS_token_state

    def forward(self, batch: dict[str, Any]):
        output = self.get_features(batch)
        if self.normalise_features:
            output = normalise(output)
        return output

    @property
    def feat_dim(self):
        return self.model.config.hidden_size
