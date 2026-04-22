"""data_cleaning.py"""
import os
import numpy as np
import pandas as pd

NYC_LAT = (40.4774, 41.0176)
NYC_LON = (-74.2591, -73.6004)


def clean(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    original = len(df)
    log = []

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["fare_amount"]     = pd.to_numeric(df["fare_amount"],     errors="coerce")
    df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce")
    for c in ["pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    critical = ["fare_amount","pickup_datetime","pickup_latitude",
                "pickup_longitude","dropoff_latitude","dropoff_longitude"]
    n = len(df); df = df.dropna(subset=critical); log.append(("drop_nulls", n - len(df)))

    median_pax = df["passenger_count"].median()
    df["passenger_count"] = df["passenger_count"].fillna(median_pax).clip(1, 6).astype(int)

    fare_upper = df["fare_amount"].quantile(0.999)
    n = len(df); df = df[(df["fare_amount"] >= 2.50) & (df["fare_amount"] <= fare_upper)]
    log.append(("fare_filter", n - len(df)))

    n = len(df)
    geo = (df["pickup_latitude"].between(*NYC_LAT)  &
           df["pickup_longitude"].between(*NYC_LON) &
           df["dropoff_latitude"].between(*NYC_LAT) &
           df["dropoff_longitude"].between(*NYC_LON))
    df = df[geo]; log.append(("geo_filter", n - len(df)))

    n = len(df)
    same = ((df["pickup_latitude"]  == df["dropoff_latitude"]) &
            (df["pickup_longitude"] == df["dropoff_longitude"]))
    df = df[~same]; log.append(("zero_dist", n - len(df)))

    if "key" in df.columns:
        n = len(df); df = df.drop_duplicates(subset="key"); log.append(("dedup", n - len(df)))

    df = df.reset_index(drop=True)
    if verbose:
        total = sum(v for _, v in log)
        print(f"[clean] {original:,} -> {len(df):,} rows  (dropped {total:,}  "
              f"retained {len(df)/original*100:.1f}%)")
    return df


def run(raw="data/raw/uber_fares.csv", out="data/processed/uber_clean.csv"):
    df = pd.read_csv(raw)
    df = clean(df)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    run()
