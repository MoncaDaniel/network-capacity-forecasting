import os
import pandas as pd
import plotly.express as px

BASELINE = "reports/capacity_risk_horizons.csv"
WHATIF = "reports/capacity_risk_whatif_users_1p2.csv"
OUTDIR = "reports/figures"

def load():
    base = pd.read_csv(BASELINE)
    wi = pd.read_csv(WHATIF)
    return base, wi

def agg_region(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("region", as_index=False)
          .agg(
              max_risk=("p_worst", "max"),
              mean_risk=("p_worst", "mean"),
              high_cells=("risk_level", lambda s: int((s == "HIGH").sum())),
              n_cells=("cell_id", "count"),
          )
    )

def plot_baseline_risk(base):
    agg = agg_region(base).sort_values("max_risk", ascending=False)

    fig = px.bar(
        agg,
        x="region",
        y="max_risk",
        title="Baseline — worst-case saturation risk by region",
        hover_data=["mean_risk", "high_cells", "n_cells"]
    )
    fig.update_layout(
        yaxis_title="Worst-case saturation probability",
        xaxis_title="Region"
    )
    fig.write_html(os.path.join(OUTDIR, "baseline_risk_by_region.html"), include_plotlyjs="cdn")

def plot_delta(base, wi):
    b = agg_region(base).rename(columns={"max_risk": "base"})
    w = agg_region(wi).rename(columns={"max_risk": "whatif"})

    m = b.merge(w, on="region", suffixes=("_base", "_wi"))
    m["delta"] = m["whatif"] - m["base"]
    m = m.sort_values("delta", ascending=False)

    fig = px.bar(
        m,
        x="region",
        y="delta",
        title="What-if +20% users — increase in saturation risk by region",
        hover_data=["base", "whatif"]
    )
    fig.update_layout(
        yaxis_title="Δ saturation probability",
        xaxis_title="Region"
    )
    fig.write_html(os.path.join(OUTDIR, "delta_risk_by_region.html"), include_plotlyjs="cdn")

def plot_heatmap(base, wi):
    b = agg_region(base).rename(columns={"max_risk": "baseline"})
    w = agg_region(wi).rename(columns={"max_risk": "what_if"})

    m = b.merge(w, on="region")
    heat = m.set_index("region")[["baseline", "what_if"]]

    fig = px.imshow(
        heat,
        title="Capacity risk comparison — baseline vs what-if (+20% users)",
        aspect="auto",
        text_auto=".2f"
    )
    fig.write_html(os.path.join(OUTDIR, "risk_comparison_heatmap.html"), include_plotlyjs="cdn")

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    base, wi = load()

    plot_baseline_risk(base)
    plot_delta(base, wi)
    plot_heatmap(base, wi)

    print("OK ✅ Professional reports generated in reports/figures/")
    print("- baseline_risk_by_region.html")
    print("- delta_risk_by_region.html")
    print("- risk_comparison_heatmap.html")

if __name__ == "__main__":
    main()
