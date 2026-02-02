import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df

def add_saturation_label(df: pd.DataFrame, threshold_mbps: float) -> pd.DataFrame:
    df = df.copy()
    df["is_saturated"] = (df["traffic_mbps"] >= threshold_mbps).astype(int)
    return df
