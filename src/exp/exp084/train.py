import argparse
import gc
import importlib
import multiprocessing as mp
import pathlib
import pprint
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import pydantic
import torch
import torch.nn as nn
import wandb
from sklearn import model_selection, preprocessing
from timm import utils as timm_utils
from torch.utils import data as torch_data
from tqdm.auto import tqdm

from src import constants, log, metrics, optim, train_tools, utils, visualize

# from typing import cast
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logger = log.get_root_logger()

EXP_NO = "exp084"
DESCRIPTION = """
006
    + seq2seq model (LeapFormer, 007)
    + num_encoder_layers=12 (008)
    + model_dim=25, seq_len=60, out_dim=368, seq_model_dim=128 (009)
    + なんかモデル色々
    + headの仕組みの変更 (012)
    + headの仕組みの変更 (015)
    + headに入力と埋め込みを受け取って出力するrevise用のCNNBlockを追加 (016)
    + more data 1000000 * 4 (017)
    + use sampled data (019)
    + model: Unet1D (020)
    + CNN block head (021)
    + Unet w/ ResBlock (024)
    + change head (025)
    + replace middle to CNNResidualBlock (029)
    + data size 6000000, use 4000000 (030)
    + SE Module (031)
    + use 6000000 (032)
    + early stopping (033)
    + Train w/ OLD WEIGHTS and then predict divided by the weights and multiply the new weights (034)
    + kernel_size=7 (41)
    + in_chnas=(25, 64, 128, 256, 512, 1024) (042)
    + valid_size=650000 (043)
    + random feature order augmentation (045)
    + kernel_size=3 in UNet1D (058)
    + LSTM Neck (060)
    - random feature order augmentation (061)
    + decoderのconvをSEResidualBlockに変更 (062)
    + LeapUnet1dのconvをSEResidualBlockに変更 (063)
    + UnetEncoderのres_convの前後にskip connectionを追加 (069)
    + UnetDecoderのres_convの前後にskip connectionを追加 (071)
    + AdaptiveAvgPool1dを追加 (074)
    + Unetのcnnをnn.Conv1dに変更 (075)
    + skip connnectionをunetの後に追加 (079)
    + unet_in_chans=(25, 128, 256, 512, 1024, 2048) (081)
    + use train_sampled_8000000.parquet (082)k
"""
CALLED_TIME = log.get_called_time()


models = importlib.import_module(f"src.exp.{EXP_NO}.models")


class Config(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str = pydantic.Field(default=EXP_NO, description="experiment name")
    description: str = pydantic.Field(default=DESCRIPTION, description="experiment description")

    # -- General
    is_debug: bool = pydantic.Field(default=False, description="debug mode")
    root_dir: pathlib.Path = pydantic.Field(default=constants.ROOT, description="root directory")
    input_dir: pathlib.Path = pydantic.Field(default=constants.INPUT_DIR, description="input directory")
    output_dir: pathlib.Path = pydantic.Field(default=constants.OUTPUT_DIR / EXP_NO, description="output directory")
    data_dir: pathlib.Path = pydantic.Field(default=constants.DATA_DIR, description="data directory")
    device: torch.device = pydantic.Field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), description="device"
    )
    seed: int = pydantic.Field(default=1995, description="seed")
    # -- Model
    unet_in_chans: tuple[int, int, int, int, int, int] = pydantic.Field(
        # default_factory=lambda: (25, 32, 64, 128, 256, 512)
        default_factory=lambda: (25, 128, 256, 512, 1024, 2048)
    )
    # unet_in_chans: int = pydantic.Field(default_factory=lambda: (25, 64, 128, 256, 512, 1024))

    # -- Data
    n_fold: int = pydantic.Field(default=5, description="number of folds")
    # train_data: pathlib.Path = pydantic.Field(default=constants.DATA_DIR / "train.csv")
    # train_data: pathlib.Path = pydantic.Field(default=constants.INPUT_DIR / "train_sampled_4000000.parquet")
    # train_data: pathlib.Path = pydantic.Field(default=constants.INPUT_DIR / "train_sampled_6000000.parquet")
    train_data: pathlib.Path = pydantic.Field(default=constants.INPUT_DIR / "train_sampled_8000000.parquet")
    n_rows: int = pydantic.Field(default=int(1_000_000 * 7.3), description="number of rows to load")
    test_data: pathlib.Path = pydantic.Field(default=constants.DATA_DIR / "test.csv")

    # -- Train
    optimizer_name: str = pydantic.Field(default="AdamW", description="optimizer name")
    optimizer_params: dict[str, float] = pydantic.Field(
        default_factory=lambda: {"lr": 1e-3, "weight_decay": 1e-2, "eps": 1e-8}
    )
    scheduler_name: str = pydantic.Field(default="CosineLRScheduler", description="scheduler name")
    scheduler_params: dict[str, float] = pydantic.Field(
        default_factory=lambda: {
            "t_initial": 50,
            "lr_min": 1e-7,
            "warmup_prefix": True,
            "warmup_t": 0,
            "warmup_lr_init": 1e-6,
            "cycle_limit": 1,
        }
    )
    n_epochs: int = pydantic.Field(default=50, description="number of epochs")
    log_interval: int = pydantic.Field(default=1, description="log interval")
    train_batch_size: int = pydantic.Field(default=1024 * 2, description="train batch size")

    # -- Valid
    valid_batch_size: int = pydantic.Field(default=1024 * 2, description="valid batch size")


# =============================================================================
# Dataset
# =============================================================================
X_MEAN, X_STD, Y_MEAN, Y_STD = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])


def calc_mean_std(x: torch.Tensor, y: torch.Tensor, err: float = 1e-6) -> None:
    """
    Calculate and set the global mean and standard deviation for the dataset features and targets.
    """

    global X_MEAN, X_STD, Y_MEAN, Y_STD

    X_MEAN = torch.mean(x, 0)
    X_STD = torch.maximum(torch.std(x, 0), torch.tensor(err))

    Y_MEAN = y.mean(0)
    Y_STD = torch.maximum(torch.sqrt(torch.mean(torch.pow(y, 2), 0)), torch.tensor(err))

    print(f"{X_MEAN.dtype = } {X_STD.dtype = } {Y_MEAN.dtype = } {Y_STD.dtype = }")


class LeapDataset(torch_data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        x_features: list[str],
        y_features: list[str],
    ) -> None:
        super().__init__()

        # x = torch.from_numpy(data[x_features].to_numpy())
        x = data[x_features]
        self.x = x

        w = torch.tensor(constants.OLD_TARGET_WEIGHTS)
        y = torch.from_numpy(data[y_features].to_numpy())
        self.y = y * w

        self.sample_ids = data["sample_id"].to_numpy()

        logger.info(f"{self.y.min() = } {y.min() = }")
        logger.info(f"{self.x.shape = } {self.y.shape = }")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, npt.NDArray[np.int64]]:
        x = torch.from_numpy(self.x.iloc[idx].to_numpy())
        y = self.y[idx]
        sample_id = self.sample_ids[idx]

        assert len(X_MEAN) > 0, "Mean and std are not calculated yet"

        # scale
        x = (x - X_MEAN) / X_STD
        y = (y - Y_MEAN) / Y_STD

        x = x.to(torch.float32)
        y = y.to(torch.float32)

        return x, y, sample_id

    def __len__(self) -> int:
        return len(self.y)


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in tqdm(df.columns, total=len(df.columns)):
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df.loc[:, col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df.loc[:, col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df.loc[:, col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df.loc[:, col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df.loc[:, col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df.loc[:, col] = df[col].astype(np.float32)
                else:
                    df.loc[:, col] = df[col].astype(np.float64)
        else:
            df.loc[:, col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def reduce_memory_usage_pl(df: pl.DataFrame, name: str) -> pl.DataFrame:
    print(f"Memory usage of dataframe {name} is {round(df.estimated_size('mb'), 2)} MB")
    Numeric_Int_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
    Numeric_Float_types = [pl.Float32, pl.Float64]
    float32_tiny = np.finfo(np.float32).tiny.astype(np.float64)
    float32_min = np.finfo(np.float32).min.astype(np.float64)
    float32_max = np.finfo(np.float32).max.astype(np.float64)
    for col in tqdm(df.columns, total=len(df.columns)):
        if col in ["sample_id"]:
            continue
        col_type = df[col].dtype
        c_min = df[col].to_numpy().min()
        c_max = df[col].to_numpy().max()
        if col_type in Numeric_Int_types:
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:  # type: ignore
                df = df.with_columns(df[col].cast(pl.Int8))
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:  # type: ignore
                df = df.with_columns(df[col].cast(pl.Int16))
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:  # type: ignore
                df = df.with_columns(df[col].cast(pl.Int32))
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:  # type: ignore
                df = df.with_columns(df[col].cast(pl.Int64))
        elif col_type in Numeric_Float_types:
            if (
                (float32_min < c_min < float32_max)
                and (float32_min < c_max < float32_max)
                and (abs(c_min) > float32_tiny)
                and (abs(c_max) > float32_tiny)
            ):
                # print(f"{col} => {col_type}")
                # print(f"{c_min = } / {c_min.dtype} : {c_max = } / {c_max.dtype}")
                # print(f"{float32_min = } / {float32_min.dtype} : {float32_max = } / {float32_max.dtype}")
                df = df.with_columns(df[col].cast(pl.Float32).alias(col))
            else:
                pass
        elif col_type == pl.Utf8:
            df = df.with_columns(df[col].cast(pl.Categorical))
        else:
            pass
    print(f"Memory usage of dataframe {name} became {round(df.estimated_size('mb'), 2)} MB")
    return df


def _load_train_valid_data(
    cfg: Config, file_path: pathlib.Path, n_rows: int, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    file_path = cfg.train_data
    n_rows = cfg.n_rows if not cfg.is_debug else 5000
    logger.info(f"Load data from {file_path}")
    if file_path.suffix == ".parquet":
        data = pl.read_parquet(file_path, n_rows=n_rows, low_memory=True)
    elif file_path.suffix == ".csv":
        data = pl.read_csv(file_path, n_rows=n_rows)
    else:
        raise ValueError("file format is not supported")

    print(data.describe())

    # data = data.to_pandas()
    # data.loc[:, constants.FEATURE_NAMES] = reduce_mem_usage(data[constants.FEATURE_NAMES])
    # data.loc[:, constants.TARGET_NAMES] = reduce_mem_usage(data[constants.TARGET_NAMES])
    # data = data.shrink_to_fit(in_place=False)
    data = reduce_memory_usage_pl(data, "data")
    data = data.to_pandas()
    print(data.info())

    utils.seed_everything(cfg.seed + fold)
    # data = data.to_pandas(use_pyarrow_extension_array=True)
    logger.info(f"{data.shape = }")
    valid_size = 650_000
    if cfg.is_debug:
        valid_size = len(data) // 2
    # n_folds = len(data) // valid_size
    # if cfg.is_debug:
    #     n_folds = 2
    # logger.info(f"n_folds: {n_folds}")
    # kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=cfg.seed)
    # for i, (_train_idx, valid_idx) in enumerate(kf.split(data)):
    #     data.loc[valid_idx, "fold"] = i
    train_data, valid_data = model_selection.train_test_split(data, test_size=valid_size, random_state=cfg.seed)

    # data = pl.from_pandas(data)
    # train_data = data.filter(pl.col("fold") != fold)
    # valid_data = data.filter(pl.col("fold") == fold)
    logger.info(f"train size: {len(train_data)} valid size: {len(valid_data)}")
    return train_data, valid_data


def init_dataloader(
    cfg: Config, fold: int, num_workers: int = 16
) -> tuple[torch_data.DataLoader, torch_data.DataLoader]:
    if mp.cpu_count() < num_workers:
        num_workers = mp.cpu_count()
        logger.info(f"num_workers is set to {num_workers}")
    elif cfg.is_debug:
        num_workers = 0

    train_data, valid_data = _load_train_valid_data(cfg, cfg.train_data, cfg.n_rows, fold)

    train_ds = LeapDataset(
        data=train_data,
        x_features=constants.FEATURE_NAMES,
        y_features=constants.TARGET_NAMES,
    )
    valid_ds = LeapDataset(
        data=valid_data,
        x_features=constants.FEATURE_NAMES,
        y_features=constants.TARGET_NAMES,
    )
    calc_mean_std(torch.from_numpy(train_ds.x.to_numpy()).to(torch.float64, non_blocking=True), train_ds.y)  # type: ignore

    logger.info(f"train size: {len(train_ds)} valid size: {len(valid_ds)}")

    train_dl = torch_data.DataLoader(
        dataset=train_ds,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=lambda _: utils.seed_everything(42),
        persistent_workers=True if num_workers > 0 else False,
    )

    valid_loader = torch_data.DataLoader(
        dataset=valid_ds,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=lambda _: utils.seed_everything(42),
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_dl, valid_loader


# =============================================================================
# Train
# =============================================================================


def _get_random_feature_idx(n_split: int = 3, feature_size: int = 60) -> npt.NDArray[np.int32]:
    """
    n_splitで分割して、それぞれの中でランダムに一つindex選ぶ

    Args:
        n_split: 分割数

    Returns:
        n_split分の分割された中から選ばれたindex. max(index) = feature_size.
    """
    index_grps = np.array_split(np.arange(feature_size), n_split)
    return np.array([np.random.choice(idx_grp) for idx_grp in index_grps])


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    ema_model: timm_utils.ModelEmaV2,
    criterion: nn.Module,
    loader: torch_data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: optim.Scheduler,
    device: torch.device,
) -> tuple[float, float]:
    """
    Returns:
        train_avg_loss, lr
    """
    model = model.train()
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Train", dynamic_ncols=True)
    loss_meter = train_tools.AverageMeter("train/loss")
    for _batch_idx, batch in pbar:
        ema_model.update(model)
        x, y, _ = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # if np.random.rand() < 0.5:
        #     random_idx = _get_random_feature_idx()
        # else:
        #     random_idx = None

        output = model(x)
        y_pred = output

        loss = criterion(y_pred, y)

        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(epoch=epoch)

        loss = loss.detach().cpu().item()
        loss_meter.update(loss)
        pbar.set_postfix_str(f"Loss:{loss_meter.avg:.4f}")

    return loss_meter.avg, optimizer.param_groups[0]["lr"]


# =============================================================================
# Valid
# =============================================================================
def valid_one_epoch(
    model: nn.Module,
    loader: torch_data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, pl.DataFrame, npt.NDArray]:
    """
    Returns:
        valid_loss, valid_score, valid_oof
    """
    model = model.eval()
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Valid", dynamic_ncols=True)
    loss_meter = train_tools.AverageMeter("valid/loss")
    oof: list[pl.DataFrame] = []

    replace_from_col_idx = [constants.TARGET_NAMES.index(col) for col in constants.REPLACE_FROM]
    replace_to_col_idx = [constants.FEATURE_NAMES.index(col) for col in constants.REPLACE_TO]
    replace_w = torch.tensor([constants.TARGET_WEIGHTS[i] for i in replace_from_col_idx])

    y_trues, y_preds = [], []
    for batch_idx, batch in pbar:
        x, y, sample_ids = batch
        x = x.to(device, non_blocking=True)
        # with torch.inference_mode():
        with torch.no_grad():
            output = model(x)

        y = y.detach().cpu()
        y_pred = output.detach().cpu()

        replace_x = x[:, replace_to_col_idx].detach().cpu()
        static_pred = -replace_x * replace_w / 1200
        y_pred[:, replace_from_col_idx] = static_pred

        loss = criterion(y_pred, y)
        loss_meter.update(loss.item())

        y_preds.append(y_pred)
        y_trues.append(y)

        oof.append(
            train_tools.make_oof(
                x.detach().cpu().numpy(),
                y.numpy(),
                y_pred.numpy(),
                constants.FEATURE_NAMES,
                constants.TARGET_NAMES,
                id=list(sample_ids),
            )
        )
        if batch_idx % 20 == 0:
            pbar.set_postfix_str(f"Loss:{loss_meter.avg:.4f}")

    old_w = torch.tensor(constants.OLD_TARGET_WEIGHTS)
    old_w = torch.where(old_w == 0.0, torch.tensor(1.0), old_w)
    w = torch.tensor(constants.TARGET_WEIGHTS)

    y_preds = torch.concat(y_preds, dim=0)
    y_preds = (y_preds * Y_STD) + Y_MEAN
    y_preds = y_preds / old_w
    y_preds = y_preds * w

    y_trues = torch.concat(y_trues, dim=0)
    y_trues = (y_trues * Y_STD) + Y_MEAN
    y_trues = y_trues / old_w
    y_trues = y_trues * w

    valid_score, valid_scores = metrics.r2_score(y_pred=y_preds, y_true=y_trues)
    oof_df = pl.concat(oof) if len(oof) > 0 else pl.DataFrame()
    return loss_meter.avg, valid_score, oof_df, valid_scores


def main() -> None:
    # =============================================================================
    # Setup
    # =============================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--full", default=False, action="store_true")
    args = parser.parse_args()

    if args.debug:
        cfg = Config(is_debug=True, n_epochs=1, train_batch_size=32, valid_batch_size=32)
    else:
        cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    log.attach_file_handler(logger, str(cfg.output_dir / "train.log"))
    logger.info(f"Exp: {cfg.name}, DESC: {cfg.description}, COMMIT_HASH: {utils.get_commit_hash_head()}")
    logger.info(pprint.pformat(cfg.model_dump()))
    # =============================================================================
    # TrainLoop
    # =============================================================================
    for fold in range(cfg.n_fold):
        if not args.full and fold > 0:
            break

        logger.info(f"Start fold: {fold}")
        utils.seed_everything(cfg.seed + fold)
        if cfg.is_debug:
            run = None
        else:
            run = wandb.init(
                project=constants.COMPE_NAME,
                name=f"{cfg.name}_{fold}",
                config=cfg.model_dump(),
                reinit=True,
                group=cfg.name.split("-")[0],
                dir="./src",
            )

        # compile_mode = "max-autotune"
        compile_mode = "default"
        model = models.LeapUnet1D(cfg.unet_in_chans)
        model = model.to(cfg.device, non_blocking=True)
        logger.info(f"Model: \n{model}")
        model = cast(nn.Module, torch.compile(model, mode=compile_mode, dynamic=False))
        ema_model = timm_utils.ModelEmaV2(model, decay=0.998).to(cfg.device, non_blocking=True)
        ema_model = cast(timm_utils.ModelEmaV2, torch.compile(ema_model, mode=compile_mode, dynamic=False))

        train_loader, valid_loader = init_dataloader(cfg, fold)

        optimizer = optim.init_optimizer(cfg.optimizer_name, model, cfg.optimizer_params)
        scheduler = optim.init_scheduler(cfg.scheduler_name, optimizer, cfg.scheduler_params)
        criterion = nn.MSELoss()
        criterion_train = nn.HuberLoss(delta=1.0)
        metrics_monitor = train_tools.MetricsMonitor(
            metrics=["epoch", "train/loss", "lr", "valid/loss", "valid/score"]
        )
        early_stopping = train_tools.EarlyStopping(patience=10, direction="max")

        best_score, best_oof, best_valid_loss = 0.0, pl.DataFrame(), float("inf")
        # best_valid_scores = np.zeros(len(constants.TARGET_NAMES))
        for epoch in range(cfg.n_epochs):
            logger.info(f"Start epoch: {epoch}")
            utils.seed_everything(cfg.seed + fold)
            train_avg_loss, lr = train_one_epoch(
                epoch=epoch,
                model=model,
                ema_model=ema_model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion_train,
                device=cfg.device,
            )
            valid_avg_loss, valid_score, valid_oof, valid_scores = valid_one_epoch(
                model=model, loader=valid_loader, criterion=criterion, device=cfg.device
            )

            if valid_score > best_score:
                best_oof = valid_oof
                best_score = valid_score
                best_valid_loss = valid_avg_loss
                # best_valid_scores = valid_scores

            if run:
                metric_map = {
                    "epoch": epoch,
                    "train/loss": train_avg_loss,
                    "lr": lr,
                    "valid/loss": valid_avg_loss,
                    "valid/score": valid_score,
                }
                metrics_monitor.update(metric_map)

                for key, score in zip(constants.TARGET_NAMES, valid_scores):
                    metric_map[f"valid/score_{key}"] = float(score)
                wandb.log(metric_map)
                if epoch % cfg.log_interval == 0:
                    metrics_monitor.show()

            early_stopping.check(valid_score, model, cfg.output_dir / f"best_model_{fold}.pth")
            if early_stopping.is_early_stop:
                break

        # -- Save Results
        logger.info(f"{fold} fold Best Score: {best_score} Best Valid Loss: {best_valid_loss}")
        best_oof.write_parquet(cfg.output_dir / f"oof_{fold}.parquet")
        metrics_monitor.save(cfg.output_dir / f"metrics_{fold}.csv", fold=fold)

        print("Make plots of R2 metrics and risk")
        best_oof_y_true = best_oof[constants.TARGET_NAMES]
        best_oof_y_pred = best_oof[[f"{name}_pred" for name in constants.TARGET_NAMES]]
        visualize.plot_r2_metrics_and_risk(
            y_preds=best_oof_y_pred.to_numpy(),
            y_true=best_oof_y_true.to_numpy(),
            save_dir=cfg.output_dir,
            exp_no=cfg.name,
        )

        if hasattr(model, "_orig_mod"):
            state_dict = model._orig_mod.state_dict()
        else:
            state_dict = model.state_dict()
        weight_fp = cfg.output_dir / f"last_model_{fold}.pth"
        logger.info(f"Save weight: {weight_fp}")

        torch.save(state_dict, weight_fp)
        np.save(
            cfg.output_dir / f"stats_fold{fold}.npy",
            np.array({"X_MEAN": X_MEAN, "X_STD": X_STD, "Y_MEAN": Y_MEAN, "Y_STD": Y_STD}),
        )

        del model, ema_model, optimizer, scheduler, criterion, train_loader, valid_loader
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("End of Training")
        if run is not None:
            run.finish()

        if cfg.is_debug:
            logger.info("Break Debug")
            break


if __name__ == "__main__":
    main()
