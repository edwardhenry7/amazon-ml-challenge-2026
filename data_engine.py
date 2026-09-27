import polars as pl
import os

def build_engine():
    print("Initializing Data Engine...")
    os.makedirs("data", exist_ok=True)

    # 1. Load official training data (Strict Tab Separator)
    print("Loading Official Train Data...")
    s1_train = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
    s2_train = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
    s3_train = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")

    # 2. Load Sanjana's Synthetic French Data
    print("Loading Synthetic French Data...")
    s1_syn = pl.read_csv("synthetic/source1_synthetic.tsv", separator="\t")
    s2_syn = pl.read_csv("synthetic/source2_synthetic.tsv", separator="\t")
    s3_syn = pl.read_csv("synthetic/source3_synthetic.tsv", separator="\t")

    # 3. Concatenate to create the Master Dataset
    s1_master = pl.concat([s1_train, s1_syn])
    s2_master = pl.concat([s2_train, s2_syn])
    s3_master = pl.concat([s3_train, s3_syn])

    # 4. Export as highly optimized Parquet files for Bharath's ML script
    print("Exporting to fast Parquet format...")
    s1_master.write_parquet("data/s1_master.parquet")
    s2_master.write_parquet("data/s2_master.parquet")
    s3_master.write_parquet("data/s3_master.parquet")
    print("Data Engine Complete. Master files saved to /data/")

if __name__ == "__main__":
    build_engine()