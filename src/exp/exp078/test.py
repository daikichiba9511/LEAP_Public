import importlib

import numpy as np
import numpy.typing as npt
import polars as pl
import torch
import torch.nn as nn

from src import constants
from src.exp.exp078 import train


def pred_fn(
    model: nn.Module, df_test: pl.DataFrame, device: torch.device, stats: dict[str, torch.Tensor], batch_size: int
) -> npt.NDArray[np.float64]:
    """
    Generate predictions using the trained model on the test data.
    """
    x_mean, x_std = stats["X_MEAN"], stats["X_STD"]
    y_mean, y_std = stats["Y_MEAN"], stats["Y_STD"]

    model.eval()

    x_test = df_test[constants.FEATURE_NAMES]
    x_test = x_test.to_numpy()
    x_test = torch.from_numpy(x_test)
    print(f"x_test: {x_test.shape}, {x_test.dtype}")

    x_test = (x_test - x_mean) / x_std

    x_test = x_test.to(torch.float32)
    x_test = x_test.to(device=device, non_blocking=True)

    y_pred = []
    for i in range(0, x_test.shape[0], batch_size):
        with torch.inference_mode():
            out = model(x_test[i : i + batch_size])
            y_pred.append(out)

    y_pred = torch.cat(y_pred, dim=0)
    y_pred = y_pred.cpu()
    y_pred = y_pred.to(torch.float64)

    y_pred[:, y_std < (1.1 * 1e-6)] = 0
    y_pred = (y_pred * y_std) + y_mean

    new_w = torch.tensor(constants.TARGET_WEIGHTS)
    old_w = torch.tensor(constants.OLD_TARGET_WEIGHTS)
    old_w = torch.where(old_w == 0, torch.tensor(1.0), old_w)
    y_pred = y_pred / old_w
    y_pred = y_pred * new_w

    y_pred = y_pred.detach()
    y_pred = y_pred.cpu()
    y_pred = y_pred.numpy()

    return y_pred


def main() -> None:
    exp_ver = __file__.split(".")[0].split("/")[-2]
    cfg = train.Config()
    if cfg.name != exp_ver:
        raise ValueError(f"Config exp_ver mismatch: {cfg.name} != {exp_ver}")
    models = importlib.import_module(f"src.exp.{exp_ver}.models")
    model = models.LeapUnet1D(cfg.unet_in_chans)
    model = model.to(cfg.device, non_blocking=True)
    print(model)

    # weight_fp = cfg.output_dir / "last_model_0.pth"
    weight_fp = cfg.output_dir / "best_model_0.pth"
    print(f"Load weight: {weight_fp}")
    _weights = torch.load(weight_fp, map_location=cfg.device)
    weights = {}
    for k, v in _weights.items():
        if "_orig_mod" in k:
            k = k.replace("_orig_mod.", "")
        weights[k] = v
    print(model.load_state_dict(weights))

    df_test = pl.read_csv(cfg.test_data)
    df_test = df_test.to_pandas()
    df_test = df_test.set_index("sample_id")

    df_subm = pl.read_csv(cfg.data_dir / "sample_submission.csv")
    df_subm = df_subm.to_pandas()
    df_subm = df_subm.set_index("sample_id")

    stats = np.load(cfg.output_dir / "stats_fold0.npy", allow_pickle=True).item()

    static_pred = -df_test[constants.REPLACE_TO].to_numpy() * df_subm[constants.REPLACE_FROM].to_numpy() / 1200

    preds = pred_fn(model, pl.from_pandas(df_test), cfg.device, stats, cfg.valid_batch_size)
    assert preds.shape[0] == df_test.shape[0]

    df_subm.loc[df_test.index, constants.TARGET_NAMES] = preds
    df_subm[constants.REPLACE_FROM] = static_pred

    df_subm = df_subm.reset_index()
    df_subm = df_subm[["sample_id"] + constants.TARGET_NAMES]

    df_subm = pl.from_pandas(df_subm)
    print(df_subm)
    df_subm.write_parquet(cfg.output_dir / "submission.parquet")


if __name__ == "__main__":
    main()
