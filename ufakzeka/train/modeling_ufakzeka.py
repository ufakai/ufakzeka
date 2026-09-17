"""Hugging Face model class for ufakzeka checkpoints.

The network is a Qwen3 style decoder (RMSNorm, SwiGLU, RoPE, GQA, QK-Norm, tied
embeddings). The one difference is logit soft-capping, logits = 30 * tanh(logits / 30),
applied during training and required at inference for matching results. This file is
shipped inside the model folder and loaded with trust_remote_code=True."""

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


class UfakzekaConfig(Qwen3Config):
    model_type = "ufakzeka"

    def __init__(self, final_logit_softcapping: float = 30.0, **kwargs):
        super().__init__(**kwargs)
        self.final_logit_softcapping = final_logit_softcapping


class UfakzekaForCausalLM(Qwen3ForCausalLM):
    config_class = UfakzekaConfig

    def forward(self, *args, **kwargs):
        labels = kwargs.pop("labels", None)
        out = super().forward(*args, **kwargs)
        cap = self.config.final_logit_softcapping
        if cap:
            out.logits = cap * torch.tanh(out.logits.float() / cap)
        if labels is not None:
            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            out.loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        return out
