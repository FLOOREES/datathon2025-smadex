# import dask
# import dask.dataframe as dd
# dask.config.set({"dataframe.convert-string": False})

# dataset_path = "/home/pablo/Documents/datathon2025-smadex/data/train/train"

# ddf = dd.read_parquet(
#     dataset_path,
# )

# if __name__ == "__main__":
#     df_sample = ddf.head(5)
#     print(df_sample)

import dask.dataframe as dd

# Path to all parquet files inside nested datetime=... directories
path = "./data/smadex-challenge-predict-the-revenue/test/test/datetime=*/part-*.parquet"

# Read using Dask
df = dd.read_parquet(path)

# Trigger computation to show first rows
print(df.head())