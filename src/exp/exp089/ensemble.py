import pathlib

import polars as pl

pathlib.Path("output/exp089").mkdir(parents=True, exist_ok=True)

exp078_df = pl.read_parquet(pathlib.Path("output/exp078/submission.parquet"))
exp080_df = pl.read_parquet(pathlib.Path("output/exp080/submission.parquet"))
exp081_df = pl.read_parquet(pathlib.Path("output/exp081/submission.parquet"))
exp082_df = pl.read_parquet(pathlib.Path("output/exp082/submission.parquet"))
exp084_df = pl.read_parquet(pathlib.Path("output/exp084/submission.parquet"))
exp087_df = pl.read_parquet(pathlib.Path("output/exp087/submission.parquet"))

print(exp081_df)
print(exp082_df)

w = 1 / 6

sample_id = exp081_df["sample_id"].to_frame()
sub = (
    w * exp078_df.drop("sample_id")
    + w * exp080_df.drop("sample_id")
    + w * exp081_df.drop("sample_id")
    + w * exp082_df.drop("sample_id")
    + w * exp084_df.drop("sample_id")
    + w * exp087_df.drop("sample_id")
)
sub = pl.concat([sample_id, sub], how="horizontal")
print(sub)
sub.write_parquet("output/exp089/submission.parquet")
