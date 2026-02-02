import os
import pandas as pd

from .simulate import generate_synthetic_network_data
from .config import ForecastConfig
from .features import add_saturation_label
from .model_xgb import train_xgb_forecast
from .forecast import forecast_xgb_autoregressive
from .risk import estimate_residuals, window_saturation_probability, risk_level

def run_risk(df: pd.DataFrame, cfg: ForecastConfig, users_multiplier: float = 1.0) -> pd.DataFrame:
    H7 = cfg.horizon_days_short * 24
    H30 = cfg.horizon_days_long * 24

    cell_ids = sorted(df["cell_id"].unique())[:20]  # MVP: 20 cellules
    rows = []

    for cell_id in cell_ids:
        df_cell = df[df["cell_id"] == cell_id].copy()
        zone = df_cell["zone_type"].iloc[0]
        threshold = float(cfg.saturation_threshold_by_zone.get(zone, 800.0))

        model, feats, mae, valid_out = train_xgb_forecast(df, cell_id=cell_id, train_end="2025-07-01")
        residuals = estimate_residuals(valid_out["y_true"].values, valid_out["y_pred"].values)

        f7 = forecast_xgb_autoregressive(model, df_cell, feats, horizon_hours=H7)
        f30 = forecast_xgb_autoregressive(model, df_cell, feats, horizon_hours=H30)

        # What-if: proxy simple -> plus d'abonnés = plus de charge
        f7_adj = f7["y_pred"].values * users_multiplier
        f30_adj = f30["y_pred"].values * users_multiplier

        p7 = window_saturation_probability(f7_adj, residuals, threshold, n_paths=2000, seed=42)
        p30 = window_saturation_probability(f30_adj, residuals, threshold, n_paths=2000, seed=42)

        max7 = float(f7_adj.max())
        max30 = float(f30_adj.max())
        worst = max(p7, p30)

        cell_meta = df_cell.iloc[0]
        rows.append({
            "cell_id": cell_id,
            "region": cell_meta["region"],
            "zone_type": zone,
            "users_multiplier": users_multiplier,
            "mae_valid_mbps": round(mae, 2),
            "saturation_threshold_mbps": threshold,
            "max_pred_j7_mbps": round(max7, 2),
            "max_pred_j30_mbps": round(max30, 2),
            "p_saturation_j7": round(p7, 4),
            "p_saturation_j30": round(p30, 4),
            "risk_level": risk_level(worst),
            "p_worst": round(worst, 4),
        })

    out = pd.DataFrame(rows)
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNKNOWN": 3}
    out["risk_rank"] = out["risk_level"].map(order).fillna(9).astype(int)
    out = out.sort_values(["risk_rank", "p_worst"], ascending=[True, False]).drop(columns=["risk_rank"])
    return out

def main():
    cfg = ForecastConfig()
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("1) Génération data synthétique…")
    df = generate_synthetic_network_data(n_cells=60)
    df.to_parquet("data/processed/network_capacity.parquet", index=False)

    print("2) Ajout label saturation (seuil par zone)…")
    df = add_saturation_label(df, cfg.saturation_threshold_by_zone)

    print("3) Risk baseline (users x1.0)…")
    out_base = run_risk(df, cfg, users_multiplier=1.0)
    out_base.to_csv("reports/capacity_risk_horizons.csv", index=False)
    print("OK ✅ reports/capacity_risk_horizons.csv")
    print(out_base.head(10).to_string(index=False))

    print("4) What-if +20% abonnés (users x1.2)…")
    out_wi = run_risk(df, cfg, users_multiplier=1.2)
    out_wi.to_csv("reports/capacity_risk_whatif_users_1p2.csv", index=False)
    print("OK ✅ reports/capacity_risk_whatif_users_1p2.csv")
    print(out_wi.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
