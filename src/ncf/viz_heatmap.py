import pandas as pd
import plotly.express as px

def agg_region(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("region", as_index=False)
          .agg(
              max_p=("p_worst", "max"),
              mean_p=("p_worst", "mean"),
              n_cells=("cell_id", "count"),
              high_cells=("risk_level", lambda s: int((s == "HIGH").sum())),
              med_cells=("risk_level", lambda s: int((s == "MEDIUM").sum())),
          )
    )

def main():
    base = pd.read_csv("reports/capacity_risk_horizons.csv")
    wi = pd.read_csv("reports/capacity_risk_whatif_users_1p2.csv")

    base_agg = agg_region(base).rename(columns={
        "max_p": "max_p_base", "mean_p": "mean_p_base", "high_cells": "high_base", "med_cells": "med_base"
    })
    wi_agg = agg_region(wi).rename(columns={
        "max_p": "max_p_wi", "mean_p": "mean_p_wi", "high_cells": "high_wi", "med_cells": "med_wi"
    })

    merged = base_agg.merge(wi_agg, on=["region", "n_cells"], how="outer").fillna(0)
    merged["delta_max_p"] = merged["max_p_wi"] - merged["max_p_base"]
    merged["delta_high_cells"] = merged["high_wi"] - merged["high_base"]

    merged = merged.sort_values("delta_max_p", ascending=False)

    # 1) delta bar
    fig = px.bar(
        merged,
        x="region",
        y="delta_max_p",
        title="What-if +20% users — delta worst-case saturation probability (by region)",
        hover_data=["max_p_base", "max_p_wi", "high_base", "high_wi", "delta_high_cells", "n_cells"]
    )
    fig.update_layout(yaxis_title="delta max p_worst", xaxis_title="region")
    fig.write_html("reports/capacity_risk_delta_by_region.html", include_plotlyjs="cdn")

    # 2) compare heatmap (region x scenario metrics)
    heat = merged.set_index("region")[["max_p_base", "max_p_wi", "delta_max_p", "high_base", "high_wi", "delta_high_cells"]]
    fig2 = px.imshow(
        heat,
        title="Capacity Risk — baseline vs what-if (+20% users) comparison",
        aspect="auto"
    )
    fig2.write_html("reports/capacity_risk_compare_heatmap.html", include_plotlyjs="cdn")

    print("OK ✅ Comparison visualizations generated:")
    print(" - reports/capacity_risk_delta_by_region.html")
    print(" - reports/capacity_risk_compare_heatmap.html")

if __name__ == "__main__":
    main()
