from logging import getLogger
from typing import Any

import torch
from timm import scheduler as timm_scheduler

logger = getLogger(__name__)


def init_optimizer(optimizer_name: str, model: torch.nn.Module, params: dict[str, float]) -> torch.optim.Optimizer:
    """optimizerの初期化
    Args:
        optimizer_name: optimizerの名前
        model: T<:nn.Module
        params: optimizerのパラメータ

    Returns:
        optimizer
    """
    logger.info(f"Initialize optimizer: {optimizer_name} (params: {params})")

    if optimizer_name == "AdamW":
        model_params = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        prms = [
            {
                "params": [p for n, p in model_params if not any(nd in n for nd in no_decay)],
                "weight_decay": params["weight_decay"],
            },
            {
                "params": [p for n, p in model_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(params=prms, lr=params["lr"], fused=True, eps=params["eps"])
        return optimizer
    if optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=params["lr"], eps=params["eps"])
    else:
        raise NotImplementedError


Scheduler = torch.optim.lr_scheduler.LRScheduler | timm_scheduler.cosine_lr.Scheduler


def init_scheduler(scheduler_name: str, optimizer: torch.optim.Optimizer, params: dict[str, Any]) -> Scheduler:
    """schedulerの初期化

    Args:
        secheduler_name: schedulerの名前
        optimizer: T<:torch.optim.Optimizer
        params: schedulerのパラメータ

    """
    logger.info(f"Initialize optimizer: {scheduler_name} (params: {params})")

    if scheduler_name == "CosineLRScheduler":
        scheduler = timm_scheduler.CosineLRScheduler(
            optimizer,
            t_initial=params["t_initial"],
            lr_min=params["lr_min"],
            warmup_prefix=params["warmup_prefix"],
            warmup_lr_init=params["warmup_lr_init"],  # type: ignore
            warmup_t=params["warmup_t"],
            cycle_limit=params["cycle_limit"],
        )
        return scheduler
    if scheduler_name == "PolynomialLR":
        return torch.optim.lr_scheduler.PolynomialLR(optimizer, power=1.0, total_iters=params["total_steps"])
    else:
        raise NotImplementedError
