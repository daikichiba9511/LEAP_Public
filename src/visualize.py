import pathlib

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch

from src import constants, metrics


def plot_r2_metrics_and_risk(
    y_preds: npt.NDArray,
    y_true: npt.NDArray,
    exp_no: str,
    save_dir: pathlib.Path,
    show_plot: bool = False,
) -> None:
    """Plot R2 metrics and risk = max(y) / std(y)."""
    y_true_max = y_true.max(0)
    y_true_std = y_true.std(0)
    y_true_std = np.where(y_true_std == 0, 1, y_true_std)
    risk = y_true_max / y_true_std
    risk = risk / risk.max()

    r2, r2_values = metrics.r2_score(torch.from_numpy(y_preds), torch.from_numpy(y_true))
    r2_values = r2_values.clip(0, 1)

    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(r2_values, label="R2")
    ax1.plot(risk, label="Risk")

    w_zero_cols = [(i, c) for i, c in enumerate(constants.TARGET_NAMES) if constants.TARGET_WEIGHTS[i] == 0]
    # 連続してる区間をとる
    zero_ranges = []
    for i, _c in w_zero_cols:
        if not zero_ranges:
            zero_ranges.append([i, i])
        elif zero_ranges[-1][-1] + 1 == i:
            zero_ranges[-1][-1] = i
        else:
            zero_ranges.append([i, i])
    for start, end in zero_ranges:
        ax1.axvspan(start, end, color="gray", alpha=0.2)

    ax1.set_xlabel("Label Index")
    ax1.set_ylabel("R2 / Risk")
    ax1.set_title(f"{exp_no}: R2 & Risk R2_all : {r2:.4f}")

    fig.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "r2_and_risk.png")
    if show_plot:
        plt.show()


if __name__ == "__main__":
    import polars as pl

    exp_no = "exp032"
    n_rows = 1000
    print(f"{exp_no} - OOF (first {n_rows} rows)")
    oof = pl.read_csv(f"./output/{exp_no}/oof_0.csv", n_rows=n_rows)
    oof_y_trues = oof[constants.TARGET_NAMES]
    oof_y_preds = oof[[f"{name}_pred" for name in constants.TARGET_NAMES]]
    print(oof_y_trues)
    print(oof_y_preds)

    plot_r2_metrics_and_risk(
        y_preds=oof_y_preds.to_numpy(),
        y_true=oof_y_trues.to_numpy(),
        save_dir=pathlib.Path("./output/debug"),
        exp_no=exp_no,
    )
