"""feature_engineering.py"""
import os
import numpy as np
import pandas as pd

JFK = (40.6413, -73.7781)
LGA = (40.7769, -73.8740)
EWR = (40.6895, -74.1745)
NYC = (40.7128, -74.0060)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat1) / 2) ** 2
         + np.cos(phi1) * np.cos(phi2)
         * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["pickup_datetime"], utc=True)

    # Temporal
    df["hour"]  = dt.dt.hour.astype(np.int8)
    df["dow"]   = dt.dt.dayofweek.astype(np.int8)
    df["month"] = dt.dt.month.astype(np.int8)
    df["year"]  = dt.dt.year.astype(np.int16)

    df["is_rush_hour"]     = (((df["hour"]>=7)&(df["hour"]<=9))|((df["hour"]>=17)&(df["hour"]<=19))).astype(np.int8)
    df["is_night"]         = ((df["hour"]>=22)|(df["hour"]<=4)).astype(np.int8)
    df["is_weekend"]       = (df["dow"]>=5).astype(np.int8)
    df["is_holiday_season"]= df["month"].isin([11,12]).astype(np.int8)

    df["hour_sin"],  df["hour_cos"]  = np.sin(2*np.pi*df["hour"]/24),  np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"],   df["dow_cos"]   = np.sin(2*np.pi*df["dow"]/7),    np.cos(2*np.pi*df["dow"]/7)
    df["month_sin"], df["month_cos"] = np.sin(2*np.pi*df["month"]/12), np.cos(2*np.pi*df["month"]/12)

    # Spatial
    pu_lat = df["pickup_latitude"].values;   pu_lon = df["pickup_longitude"].values
    do_lat = df["dropoff_latitude"].values;  do_lon = df["dropoff_longitude"].values

    df["distance_km"]       = haversine(pu_lat, pu_lon, do_lat, do_lon).round(4)
    df["distance_manhattan"]= (np.abs(do_lat-pu_lat) + np.abs(do_lon-pu_lon)*np.cos(np.radians((pu_lat+do_lat)/2))) * 111.0
    df["bearing"]           = (np.degrees(np.arctan2(
        np.sin(np.radians(do_lon-pu_lon))*np.cos(np.radians(do_lat)),
        np.cos(np.radians(pu_lat))*np.sin(np.radians(do_lat)) -
        np.sin(np.radians(pu_lat))*np.cos(np.radians(do_lat))*np.cos(np.radians(do_lon-pu_lon))
    )) + 360) % 360
    df["lat_diff"] = (do_lat - pu_lat).round(6)
    df["lon_diff"] = (do_lon - pu_lon).round(6)

    for name, coords in [("jfk", JFK), ("lga", LGA), ("ewr", EWR), ("nyc", NYC)]:
        df[f"pu_{name}_km"] = haversine(pu_lat, pu_lon, *coords).round(4)
        df[f"do_{name}_km"] = haversine(do_lat, do_lon, *coords).round(4)

    thr = 2.5
    df["is_airport_trip"] = (
        (df["pu_jfk_km"]<thr)|(df["do_jfk_km"]<thr)|
        (df["pu_lga_km"]<thr)|(df["do_lga_km"]<thr)|
        (df["pu_ewr_km"]<thr)|(df["do_ewr_km"]<thr)
    ).astype(np.int8)

    # Interactions
    df["dist_x_rush"]    = df["distance_km"] * df["is_rush_hour"]
    df["dist_x_night"]   = df["distance_km"] * df["is_night"]
    df["dist_x_weekend"] = df["distance_km"] * df["is_weekend"]
    df["dist_x_airport"] = df["distance_km"] * df["is_airport_trip"]
    df["pax_bin"]        = pd.cut(df["passenger_count"], bins=[0,1,2,6], labels=[1,2,3]).astype(int)

    return df


FEATURE_COLS = [
    "hour","dow","month","year",
    "is_rush_hour","is_night","is_weekend","is_holiday_season",
    "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
    "distance_km","distance_manhattan","bearing","lat_diff","lon_diff",
    "pu_jfk_km","pu_lga_km","pu_ewr_km","do_jfk_km","do_lga_km","do_ewr_km",
    "pu_nyc_km","do_nyc_km","is_airport_trip",
    "dist_x_rush","dist_x_night","dist_x_weekend","dist_x_airport",
    "passenger_count","pax_bin",
]


def run(clean="data/processed/uber_clean.csv", out="data/processed/uber_features.csv"):
    df = pd.read_csv(clean)
    df = engineer(df)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[features] {df.shape[1]} columns, {len(df):,} rows -> {out}")
    return df


if __name__ == "__main__":
    run()
