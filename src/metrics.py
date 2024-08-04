import numpy as np
import torch
from sklearn import metrics


def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[float, np.ndarray]:
    """
    Calculate the R^2 (coefficient of determination) regression score.
    """

    # score = sklearn.metrics.r2_score(y_pred=y_pred, y_true=y_true, force_finite=True)
    # assert isinstance(score, float)
    # return score
    y_pred = y_pred.to(torch.float64).numpy()
    y_true = y_true.to(torch.float64).numpy()
    # r2 = metrics.r2_score(y_pred=y_pred, y_true=y_true).item()  # type: ignore
    r2_rows = np.array([metrics.r2_score(y_pred=y_pred[:, i], y_true=y_true[:, i]) for i in range(y_pred.shape[1])])
    r2 = np.mean(r2_rows.clip(0, 1))

    # sum_obs = torch.sum(y_true, dim=0)
    # mean_obs = sum_obs / y_true.shape[0]
    #
    # sum_squared_obs = torch.sum(y_true**2, dim=0)
    #
    # ss_tot = sum_squared_obs - sum_obs * mean_obs
    # ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    #
    # r2_raw = 1 - ss_res / ss_tot
    # r2 = torch.mean(r2_raw).item()

    assert isinstance(r2, float)
    return r2, r2_rows


def _test_r2_score():
    torch.manual_seed(0)

    t = torch.rand((100, 368))
    p = torch.rand((100, 368))
    print("torch")
    print("t -> t")
    r2 = r2_score(t, t)
    print(r2)

    print("p -> t")
    r2 = r2_score(p, t)
    print(r2)

    print("sklearn")
    print("t -> t")
    r2 = metrics.r2_score(t.to(torch.float64).numpy(), t.to(torch.float64).numpy())
    print(r2)

    print("p -> t")
    r2 = metrics.r2_score(t.to(torch.float64).numpy(), p.to(torch.float64).numpy(), force_finite=True)
    print(r2)


if __name__ == "__main__":
    _test_r2_score()
