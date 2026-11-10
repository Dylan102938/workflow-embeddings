"""The workflow encoder and how to build it from a run's configuration or a checkpoint."""

from __future__ import annotations

import torch

from .transformer import ModelConfig, WorkflowTransformer


def model_config(experiment, train, dataset) -> ModelConfig:
    """From config.Experiment, config.TrainConfig and the input / label dimensions of a data.WorkflowDataset."""
    return ModelConfig(
        architecture=experiment.architecture,
        d_model=experiment.d_model,
        d_head=experiment.d_head,
        n_layers=experiment.n_layers,
        n_heads=experiment.n_heads,
        dropout=experiment.dropout,
        d_label=dataset.label_keys.shape[1],
        d_screen=dataset.frames.dim,
        n_patches=dataset.frames.n_patches,
        n_key_tokens=dataset.n_key_tokens,
        d_text=dataset.text_embeddings.shape[1],
        max_events=dataset.max_events,
        screen_token_dropout=train.screen_token_dropout,
        mlp_ratio=train.mlp_ratio,
        confidence_min=train.confidence_min,
        confidence_max=train.confidence_max,
        confidence_init=train.confidence_init,
        activation_checkpointing=train.activation_checkpointing,
    )


def build_model(config: ModelConfig, text_embeddings) -> WorkflowTransformer:
    return WorkflowTransformer(config, torch.as_tensor(text_embeddings))


def load_checkpoint(path, text_embeddings, device="cpu") -> tuple[WorkflowTransformer, dict]:
    """The model in eval mode, and the checkpoint dict (model_config, experiment, step)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ModelConfig(**checkpoint["model_config"]), text_embeddings)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


__all__ = ["ModelConfig", "WorkflowTransformer", "model_config", "build_model", "load_checkpoint"]
