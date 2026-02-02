import os
import pandas as pd

from .simulate import generate_synthetic_network_data
from .config import ForecastConfig
from .features import add_saturation_label
from .model_xgb import train_xgb_forecast
from .risk import estimate_residuals, saturation_probability, risk_level

def main():
    cfg = ForecastConfig()
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("1) Génération data synthétique…")
    df = generate_synthetic_network_data(n_cells=60)
    df.to_parquet("data/processed/network_capacity.parquet", index=False)

    print("2) Ajout label saturation…")
    df = add_saturation_label(df, cfg.saturation_threshold_mbps)

    print("3) Entraînement + risk scoring (subset cellules)…")
    cell_ids = sorted(df["cell_id"].unique())[:20]  # MVP: 20 cellules
    rows = []

    for cell_id in cell_ids:
        model, feats, mae, valid_out = train_xgb_forecast(df, cell_id=cell_id, train_end="2025-07-01")

        residuals = estimate_residuals(valid_out["y_true"].values, valid_out["y_pred"].values)

        # proxy: on prend le dernier point prédit comme "future load" (MVP)
        last_pred = float(valid_out["y_pred"].iloc[-1])

        p_sat = saturation_probability(
            y_pred_point=last_pred,
            residuals=residuals,
            threshold=cfg.saturation_threshold_mbps,
            n_samples=3000,
            seed=42
        )

        cell_meta = df[df["cell_id"] == cell_id].iloc[0]
        rows.append({
            "cell_id": cell_id,
            "region": cell_meta["region"],
            "zone_type": cell_meta["zone_type"],
            "mae_valid_mbps": round(mae, 2),
            "pred_load_mbps_proxy": round(last_pred, 2),
            "saturation_threshold_mbps": cfg.saturation_threshold_mbps,
            "p_saturation": round(p_sat, 4),
            "risk_level": risk_level(p_sat),
        })

    out = pd.DataFrame(rows).sort_values(["risk_level","p_saturation"], ascending=[True, False])
    # ordre custom: HIGH > MEDIUM > LOW
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNKNOWN": 3}
    out["risk_rank"] = out["risk_level"].map(order).fillna(9).astype(int)
    out = out.sort_values(["risk_rank", "p_saturation"], ascending=[True, False]).drop(columns=["risk_rank"])

    out_path = "reports/capacity_risk_top.csv"
    out.to_csv(out_path, index=False)

    print(f"OK ✅ Rapport risk généré: {out_path}")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
