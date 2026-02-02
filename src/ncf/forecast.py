import numpy as np
import pandas as pd

from .features import add_time_features

DEFAULT_LAGS = (1, 2, 24, 48, 168)

def _build_feature_row(hist: pd.DataFrame, next_ts: pd.Timestamp, lags=DEFAULT_LAGS) -> pd.DataFrame:
    """
    Construit une ligne de features pour prédire traffic_mbps à next_ts
    à partir de l'historique (incluant traffic_mbps et users).
    """
    row = {"timestamp": next_ts}

    # lags traffic + users
    for l in lags:
        row[f"lag_{l}"] = float(hist["traffic_mbps"].iloc[-l])
        row[f"ulag_{l}"] = float(hist["users"].iloc[-l])

    tmp = pd.DataFrame([row])
    tmp = add_time_features(tmp)
    return tmp

def forecast_xgb_autoregressive(
    model,
    df_cell: pd.DataFrame,
    feats: list[str],
    horizon_hours: int,
    lags=DEFAULT_LAGS,
) -> pd.DataFrame:
    """
    Forecast auto-régressif sur horizon_hours.
    Hypothèse MVP: users restent constants au dernier niveau observé.
    (on pourra ajouter un modèle users ou un scénario what-if ensuite)
    """
    d = df_cell.sort_values("timestamp").copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])

    # historique minimal requis (max lag)
    max_lag = max(lags)
    if len(d) < max_lag + 2:
        raise ValueError(f"Not enough history for max_lag={max_lag}. Need at least {max_lag+2} rows.")

    hist = d[["timestamp", "traffic_mbps", "users"]].tail(max_lag + 1).copy()
    last_ts = hist["timestamp"].iloc[-1]
    last_users = int(hist["users"].iloc[-1])

    preds = []
    for step in range(1, horizon_hours + 1):
        ts_next = last_ts + pd.Timedelta(hours=step)

        xrow = _build_feature_row(hist, ts_next, lags=lags)

        # users future: constant MVP
        # (on ajoute quand même les lags users via hist, et on pousse une valeur users à chaque step)
        x = xrow.reindex(columns=feats, fill_value=0)

        yhat = float(model.predict(x)[0])

        preds.append({"timestamp": ts_next, "y_pred": yhat})

        # avancer l'historique: ajouter la prédiction + users constant
        hist = pd.concat(
            [hist, pd.DataFrame([{"timestamp": ts_next, "traffic_mbps": yhat, "users": last_users}])],
            ignore_index=True
        ).tail(max_lag + 1)

    return pd.DataFrame(preds)
