import joblib
import contextlib
import math
import os
import pathlib
import random
import subprocess
import polars as pl
import time
from logging import getLogger
from typing import Any, Generator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import psutil
import torch
from sklearn.preprocessing import StandardScaler

logger = getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.autograd.anomaly_mode.set_detect_anomaly(False)


def standarize(x: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    xmin = x.min(dim).values
    xmax = x.max(dim).values
    return (x - xmin) / (xmax - xmin + eps)


def standarize_np(x: npt.NDArray, axis: int = 0, eps: float = 1e-8) -> npt.NDArray:
    xmin = x.min(axis)
    xmax = x.max(axis)
    return (x - xmin) / (xmax - xmin + eps)


def rescale(x: npt.NDArray[np.float64], scaling_factor: float = 1e5) -> npt.NDArray[np.float64]:
    xmax = np.max(x)
    if xmax * scaling_factor > np.finfo(np.float64).max:
        raise ValueError("Scaling factor is too large")

    x = x * scaling_factor
    return x.astype(np.float64)


def rescale_back(x: npt.NDArray[np.float64], scaling_factor: float = 1e5) -> npt.NDArray[np.float64]:
    return x / scaling_factor


def scale_with_standard_scaler(
    df: pl.DataFrame, verbose: bool = False
) -> tuple[pl.DataFrame, dict[str, StandardScaler]]:
    """
    Args:
        df: DataFrame to scale
    Returns:
        df: Scaled DataFrame
        scalers: Dict of scalers for each column
    """
    scalers = {}
    for col in df.columns:
        if col == "sample_id":
            continue

        scaler = StandardScaler()
        x = df[col].to_numpy()
        x = x.reshape(-1, 1)
        if verbose:
            print(f" ----- {col} ----- ")
            print(f"Before: {x.min() = }, {x.max() = }")
        x = scaler.fit_transform(x)
        if verbose:
            print(f"After: {x.min() = }, {x.max() = }")
        scalers[col] = scaler
        df = df.with_columns({col: pl.Series(x.flatten())})
    return df, scalers


def save_scalers(scalers: dict[str, StandardScaler], save_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    saved_paths = {}
    for col, scaler in scalers.items():
        save_path = save_dir / f"{col}_scaler.pkl"
        with save_path.open("wb") as f:
            joblib.dump(scaler, f)

        saved_paths[col] = str(save_path)
    return saved_paths


@contextlib.contextmanager
def trace(title: str) -> Generator[None, None, None]:
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0**30
    yield
    m1 = p.memory_info().rss / 2.0**30
    delta_mem = m1 - m0
    sign = "+" if delta_mem >= 0 else "-"
    delta_mem = math.fabs(delta_mem)
    duration = time.time() - t0
    duration_min = duration / 60
    msg = f"{title}: {m1:.2f}GB ({sign}{delta_mem:.2f}GB):{duration:.4f}s ({duration_min:3f}m)"
    print(f"\n{msg}\n")


@contextlib.contextmanager
def trace_with_cuda(title: str) -> Generator[None, None, None]:
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0**30
    allocated0 = torch.cuda.memory_allocated() / 10**9
    yield
    m1 = p.memory_info().rss / 2.0**30
    delta_mem = m1 - m0
    sign = "+" if delta_mem >= 0 else "-"
    delta_mem = math.fabs(delta_mem)
    duration = time.time() - t0
    duration_min = duration / 60

    allocated1 = torch.cuda.memory_allocated() / 10**9
    delta_alloc = allocated1 - allocated0
    sign_alloc = "+" if delta_alloc >= 0 else "-"

    msg = "\n".join([
        f"{title}: => RAM:{m1:.2f}GB({sign}{delta_mem:.2f}GB) "
        f"=> VRAM:{allocated1:.2f}GB({sign_alloc}{delta_alloc:.2f}) => DUR:{duration:.4f}s({duration_min:3f}m)"
    ])
    print(f"\n{msg}\n")


def get_commit_hash_head() -> str:
    """get commit hash"""
    result = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, check=True)
    return result.stdout.decode("utf-8")[:-1]


def dbg(**kwargs: dict[Any, Any]) -> None:
    print("\n ********** DEBUG INFO ********* \n")
    print(kwargs)


def get_model_param_size(model: torch.nn.Module, only_trainable: bool = False) -> int:
    parameters = list(model.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def plot_image_pairs(
    images: Sequence[torch.Tensor | npt.NDArray[Any]],
    title: str,
    save_path: pathlib.Path | None = None,
) -> None:
    """
    Args:
        images: list of images to plot. image shape should be (C, H, W)
    """
    n_rows = len(images)
    if n_rows > 5:
        raise ValueError("Too many images to plot")

    fig, ax = plt.subplots(1, n_rows, figsize=(10, 5))
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
        ax[i].imshow(img, label=f"image_{i}")
        ax[i].set_title(f"image_{i}")

    # -- draw object overlay here

    # -- draw title & plot/save
    fig.suptitle(title)

    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(str(save_path))
        plt.close("all")
