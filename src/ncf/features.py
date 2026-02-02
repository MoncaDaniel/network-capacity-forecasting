import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df

def add_saturation_label(df: pd.DataFrame, thresholds_by_zone: dict) -> pd.DataFrame:
    df = df.copy()
    df["saturation_threshold_mbps"] = df["zone_type"].map(thresholds_by_zone).astype(float)
    df["is_saturated"] = (df["traffic_mbps"] >= df["saturation_threshold_mbps"]).astype(int)
    return df
