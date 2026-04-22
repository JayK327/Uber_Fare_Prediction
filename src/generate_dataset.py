"""
generate_dataset.py
Generates synthetic Uber NYC fare dataset matching Kaggle schema.
Use only if you don't have the real Kaggle CSV.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

NYC = dict(lat=(40.5774, 40.9176), lon=(-74.1502, -73.7002))
HOTSPOTS = [
    (40.6413, -73.7781, 0.015), (40.7769, -73.8740, 0.012),
    (40.6895, -74.1745, 0.015), (40.7580, -73.9855, 0.025),
    (40.7128, -74.0060, 0.020), (40.7282, -73.7949, 0.020),
    (40.6782, -73.9442, 0.020), (40.8448, -73.8648, 0.018),
]
AIRPORT_IDX = [0, 1, 2]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat1) / 2) ** 2
         + np.cos(phi1) * np.cos(phi2)
         * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _sample_coords(n, hotspot_prob=0.65):
    lats, lons = np.empty(n), np.empty(n)
    mask = np.random.rand(n) < hotspot_prob
    n_hot = mask.sum()
    hs_idx = np.random.randint(0, len(HOTSPOTS), n_hot)
    lats[mask]  = np.clip(np.random.normal([HOTSPOTS[i][0] for i in hs_idx],
                                            [HOTSPOTS[i][2] for i in hs_idx]), *NYC["lat"])
    lons[mask]  = np.clip(np.random.normal([HOTSPOTS[i][1] for i in hs_idx],
                                            [HOTSPOTS[i][2] for i in hs_idx]), *NYC["lon"])
    lats[~mask] = np.random.uniform(*NYC["lat"], n - n_hot)
    lons[~mask] = np.random.uniform(*NYC["lon"], n - n_hot)
    return lats, lons


def generate(n=55_000, save_path="data/raw/uber_fares.csv"):
    print(f"Generating {n:,} rows...")
    base   = datetime(2009, 1, 1)
    span_s = int((datetime(2015, 12, 31) - base).total_seconds())
    dts    = [base + timedelta(seconds=int(s)) for s in np.random.randint(0, span_s, n)]

    hour  = np.array([d.hour for d in dts], dtype=np.int8)
    dow   = np.array([d.weekday() for d in dts], dtype=np.int8)
    month = np.array([d.month for d in dts], dtype=np.int8)

    pu_lat, pu_lon = _sample_coords(n)
    do_lat, do_lon = _sample_coords(n)
    dist = haversine(pu_lat, pu_lon, do_lat, do_lon)

    surge = np.ones(n, dtype=np.float32)
    surge[((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))] *= 1.45
    surge[(hour >= 22) | (hour <= 4)] *= 1.25
    surge[dow >= 5] *= 1.15
    surge[month.isin([11, 12]) if hasattr(month, 'isin') else np.isin(month, [11, 12])] *= 1.10
    for i in AIRPORT_IDX:
        ap = haversine(pu_lat, pu_lon, HOTSPOTS[i][0], HOTSPOTS[i][1]) < 2.5
        ap |= haversine(do_lat, do_lon, HOTSPOTS[i][0], HOTSPOTS[i][1]) < 2.5
        surge[ap] *= 1.20

    pax  = np.random.choice([1,2,3,4,5,6], p=[0.55,0.22,0.11,0.07,0.03,0.02], size=n)
    fare = np.clip((2.50 + dist * 1.75) * surge + np.random.normal(0,1.5,n), 2.50, 250.0).round(2)

    df = pd.DataFrame({
        "key":               [f"{dts[i].strftime('%Y-%m-%d %H:%M:%S')}.{i}" for i in range(n)],
        "fare_amount":       fare,
        "pickup_datetime":   [d.strftime("%Y-%m-%d %H:%M:%S UTC") for d in dts],
        "pickup_longitude":  pu_lon.round(6),
        "pickup_latitude":   pu_lat.round(6),
        "dropoff_longitude": do_lon.round(6),
        "dropoff_latitude":  do_lat.round(6),
        "passenger_count":   pax.astype(np.int8),
    })
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved -> {save_path}  shape={df.shape}")
    return df


if __name__ == "__main__":
    generate()
