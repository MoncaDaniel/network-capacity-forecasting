import pandas as pd
import plotly.express as px

def main():
    df = pd.read_csv("reports/capacity_risk_top.csv")

    # agrégation: on veut un indicateur "worst-case" par région (max p_saturation)
    agg = (
        df.groupby("region", as_index=False)
          .agg(
              max_p_saturation=("p_saturation", "max"),
              mean_p_saturation=("p_saturation", "mean"),
              n_cells=("cell_id", "count"),
              high_risk_cells=("risk_level", lambda s: int((s == "HIGH").sum())),
              medium_risk_cells=("risk_level", lambda s: int((s == "MEDIUM").sum())),
          )
          .sort_values("max_p_saturation", ascending=False)
    )

    # bar chart très lisible en entretien
    fig = px.bar(
        agg,
        x="region",
        y="max_p_saturation",
        title="Capacity Risk — Worst-case saturation probability by region (max p_saturation)",
        hover_data=["mean_p_saturation", "n_cells", "high_risk_cells", "medium_risk_cells"]
    )
    fig.update_layout(yaxis_title="max p_saturation", xaxis_title="region")
    fig.write_html("reports/capacity_risk_by_region.html", include_plotlyjs="cdn")

    # matrice (heatmap) région × metric
    heat = agg.set_index("region")[["max_p_saturation", "mean_p_saturation", "high_risk_cells", "medium_risk_cells"]]
    fig2 = px.imshow(
        heat,
        title="Capacity Risk Heatmap — region x metrics",
        aspect="auto"
    )
    fig2.write_html("reports/capacity_risk_heatmap.html", include_plotlyjs="cdn")

    print("OK ✅ Heatmap générée:")
    print(" - reports/capacity_risk_by_region.html")
    print(" - reports/capacity_risk_heatmap.html")

if __name__ == "__main__":
    main()
