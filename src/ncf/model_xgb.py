import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

from .features import add_time_features

def make_supervised(df_cell: pd.DataFrame, lags=(1,2,24,48,168)) -> pd.DataFrame:
    df = df_cell.sort_values("timestamp").copy()
    for l in lags:
        df[f"lag_{l}"] = df["traffic_mbps"].shift(l)
        df[f"ulag_{l}"] = df["users"].shift(l)
    df = add_time_features(df)
    df = df.dropna()
    return df

def train_xgb_forecast(df: pd.DataFrame, cell_id: str, train_end: str):
    """
    Entraîne un XGBRegressor sur une cellule.
    Renvoie: model, feats, mae, valid_df (timestamp + y_true + y_pred)
    """
    d = df[df["cell_id"] == cell_id].copy()
    d = make_supervised(d)

    train_end = pd.to_datetime(train_end)
    train = d[d["timestamp"] < train_end]
    valid = d[d["timestamp"] >= train_end]

    feats = [c for c in train.columns if c not in ["timestamp","cell_id","region","zone_type","traffic_mbps"]]
    Xtr, ytr = train[feats], train["traffic_mbps"]
    Xva, yva = valid[feats], valid["traffic_mbps"]

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=0
    )
    model.fit(Xtr, ytr)

    pred = model.predict(Xva)
    mae = mean_absolute_error(yva, pred)

    valid_out = valid[["timestamp"]].copy()
    valid_out["y_true"] = yva.values
    valid_out["y_pred"] = pred

    return model, feats, mae, valid_out
