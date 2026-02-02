import os
import pandas as pd

from .simulate import generate_synthetic_network_data
from .config import ForecastConfig
from .features import add_saturation_label
from .model_xgb import train_xgb_forecast
from .forecast import forecast_xgb_autoregressive
from .risk import estimate_residuals, window_saturation_probability, risk_level

def main():
    cfg = ForecastConfig()
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("1) Génération data synthétique…")
    df = generate_synthetic_network_data(n_cells=60)
    df.to_parquet("data/processed/network_capacity.parquet", index=False)

    print("2) Ajout label saturation (seuil par zone)…")
    df = add_saturation_label(df, cfg.saturation_threshold_by_zone)

    print("3) Entraînement + forecast J+7/J+30 + risk scoring (subset cellules)…")
    cell_ids = sorted(df["cell_id"].unique())[:20]  # MVP: 20 cellules
    rows = []

    H7 = cfg.horizon_days_short * 24
    H30 = cfg.horizon_days_long * 24

    for cell_id in cell_ids:
        df_cell = df[df["cell_id"] == cell_id].copy()
        zone = df_cell["zone_type"].iloc[0]
        threshold = float(cfg.saturation_threshold_by_zone.get(zone, 800.0))

        model, feats, mae, valid_out = train_xgb_forecast(df, cell_id=cell_id, train_end="2025-07-01")
        residuals = estimate_residuals(valid_out["y_true"].values, valid_out["y_pred"].values)

        f7 = forecast_xgb_autoregressive(model, df_cell, feats, horizon_hours=H7)
        f30 = forecast_xgb_autoregressive(model, df_cell, feats, horizon_hours=H30)

        p7 = window_saturation_probability(
            y_pred_series=f7["y_pred"].values,
            residuals=residuals,
            threshold=threshold,
            n_paths=2000,
            seed=42
        )
        p30 = window_saturation_probability(
            y_pred_series=f30["y_pred"].values,
            residuals=residuals,
            threshold=threshold,
            n_paths=2000,
            seed=42
        )

        max7 = float(f7["y_pred"].max())
        max30 = float(f30["y_pred"].max())
        worst = max(p7, p30)

        cell_meta = df_cell.iloc[0]
        rows.append({
            "cell_id": cell_id,
            "region": cell_meta["region"],
            "zone_type": zone,
            "mae_valid_mbps": round(mae, 2),
            "saturation_threshold_mbps": threshold,
            "max_pred_j7_mbps": round(max7, 2),
            "max_pred_j30_mbps": round(max30, 2),
            "p_saturation_j7": round(p7, 4),
            "p_saturation_j30": round(p30, 4),
            "risk_level": risk_level(worst),
        })

    out = pd.DataFrame(rows)
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNKNOWN": 3}
    out["risk_rank"] = out["risk_level"].map(order).fillna(9).astype(int)

    out["p_worst"] = out[["p_saturation_j7", "p_saturation_j30"]].max(axis=1)
    out = out.sort_values(["risk_rank", "p_worst"], ascending=[True, False]).drop(columns=["risk_rank"])

    out_path = "reports/capacity_risk_horizons.csv"
    out.to_csv(out_path, index=False)

    print(f"OK ✅ Rapport horizons généré: {out_path}")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
