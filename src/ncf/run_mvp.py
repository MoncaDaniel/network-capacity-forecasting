import os
import pandas as pd

from .simulate import generate_synthetic_network_data
from .config import ForecastConfig
from .features import add_saturation_label
from .model_xgb import train_xgb_forecast

def main():
    cfg = ForecastConfig()
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("1) Génération data synthétique…")
    df = generate_synthetic_network_data(n_cells=60)
    df.to_parquet("data/processed/network_capacity.parquet", index=False)

    print("2) Ajout label saturation…")
    df = add_saturation_label(df, cfg.saturation_threshold_mbps)

    # pick une cellule au hasard pour MVP
    cell_id = df["cell_id"].iloc[0]
    print(f"3) Entraînement XGBoost sur cellule {cell_id}…")
    model, feats, mae = train_xgb_forecast(df, cell_id=cell_id, train_end="2025-07-01")
    print(f"MAE validation: {mae:.2f} Mbps")

    # TODO prochain: boucle multi-cellules + heatmap + risk scoring
    print("OK ✅ MVP entraîné (forecast baseline)")

if __name__ == "__main__":
    main()
