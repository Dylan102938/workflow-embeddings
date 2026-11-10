"""The workflow encoder (transformer.py), its event tokens (events.py) and the GRU final block (gru.py).

build_model(exp, train, data) builds the model of an experiment; checkpoints store its ModelConfig, so
load_checkpoint(path) rebuilds it without the run's configuration.
"""
from __future__ import annotations

import torch

from .transformer import ModelConfig, WorkflowTransformer


def model_config(exp, train, data) -> ModelConfig:
    """exp: config.Experiment, train: config.TrainConfig, data: data.SessionData (input / label dimensions)."""
    return ModelConfig(
        arch=exp.arch, d=exp.dm, d_head=exp.dh, n_layers=exp.nl, n_heads=exp.nh, dropout=exp.drop, qkv=exp.qkv,
        d_key=data.keys.shape[1], d_vis=data.frames.dim, n_tok=data.frames.n_tok, n_key_tokens=data.n_key_tokens,
        d_text=data.text_vecs.shape[1], e_max=int(data.ev["ev_type"].shape[1]), tok_drop=train.tok_drop,
        mlp_ratio=train.mlp_ratio, k_min=train.k_min, k_max=train.k_max, k_init=train.k_init,
        checkpointing=train.activation_checkpointing)


def build_model(cfg: ModelConfig, text_table) -> WorkflowTransformer:
    """text_table: [n_texts, d_text] frozen embeddings of every typed string (data.text_vecs)."""
    return WorkflowTransformer(cfg, torch.as_tensor(text_table))


def load_checkpoint(path, text_table, device="cpu") -> tuple[WorkflowTransformer, dict]:
    """Model (eval mode) and the checkpoint dict (model_cfg, experiment, step, ...)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ModelConfig(**state["model_cfg"]), text_table)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state


__all__ = ["ModelConfig", "WorkflowTransformer", "model_config", "build_model", "load_checkpoint"]
