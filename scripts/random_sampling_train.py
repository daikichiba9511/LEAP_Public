import polars as pl

from src import constants

train_fp = constants.DATA_DIR / "train.csv"

# sample_ratio, sample_size = 0.16, 1_500_000
# sample_ratio, sample_size = 0.4, 1_000_000 * 4
# sample_ratio, sample_size = 0.7, 1_000_000 * 6
# sample_ratio, sample_size = 0.8, 1_000_000 * 7
sample_ratio, sample_size = 0.8, 1_000_000 * 8
batch_size = 50000

reader = pl.read_csv_batched(
    train_fp,
    batch_size=batch_size,
)

batches = reader.next_batches(5)
sample_data = []
counter = 0

while batches:
    batches = pl.concat(batches)
    batches = batches.sample(fraction=sample_ratio, shuffle=True)
    sample_data.append(batches)

    counter += len(batches)
    print(counter)

    batches = reader.next_batches(5)

sample_data = pl.concat(sample_data)
print(sample_data.shape)
sample_data = sample_data.sample(sample_size, shuffle=True)
sample_data.write_parquet(f"./input/train_sampled_{sample_size}.parquet")
print(f"{sample_data.shape = }")
print(sample_data)
