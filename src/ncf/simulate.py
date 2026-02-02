import numpy as np
import pandas as pd

def generate_synthetic_network_data(
    start="2023-01-01",
    end="2025-12-31",
    n_cells=80,
    seed=42
) -> pd.DataFrame:
    """
    Génère des données horaires par cellule:
    - traffic_mbps
    - users
    - zone_type
    - event_flag
    - region
    """
    rng = np.random.default_rng(seed)
    dt = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    cells = [f"CELL_{i:04d}" for i in range(n_cells)]

    regions = ["IDF", "NAQ", "ARA", "PACA", "HDF", "OCC", "BRE", "PDL"]
    zone_types = ["urban", "suburban", "rural"]

    cell_region = {c: rng.choice(regions) for c in cells}
    cell_zone = {c: rng.choice(zone_types, p=[0.45, 0.35, 0.20]) for c in cells}

    # événements: quelques jours dans l'année (concert, match)
    event_days = pd.to_datetime(rng.choice(pd.date_range(dt.min().date(), dt.max().date(), freq="D"), size=120, replace=False))
    event_days = set(event_days.date)

    rows = []
    for cell in cells:
        zone = cell_zone[cell]
        region = cell_region[cell]

        # base selon zone
        base_users = {"urban": 450, "suburban": 260, "rural": 120}[zone]
        base_traffic = {"urban": 520, "suburban": 320, "rural": 160}[zone]  # Mbps

        # tendance lente (croissance)
        trend = np.linspace(0, 0.18, len(dt))  # +18% sur la période

        # saisonnalités
        hour = dt.hour.values
        dow = dt.dayofweek.values
        month = dt.month.values

        # profil heure: pics matin/soir
        hour_season = (
            0.25*np.sin(2*np.pi*(hour/24 - 0.2)) +
            0.35*np.exp(-0.5*((hour-20)/3.2)**2) +
            0.18*np.exp(-0.5*((hour-9)/2.8)**2)
        )

        # week-end: pattern différent
        weekend = (dow >= 5).astype(float)
        weekend_boost = 0.10*weekend

        # mois: hiver + été (tourisme)
        month_season = 0.08*np.cos(2*np.pi*(month/12 - 0.05)) + 0.06*np.exp(-0.5*((month-8)/1.4)**2)

        # événement: impulsion trafic + users sur certaines dates
        event_flag = np.array([1 if d.date() in event_days else 0 for d in dt], dtype=float)
        event_boost = event_flag * (0.22 + 0.15*rng.random(len(dt)))

        # users
        noise_u = rng.normal(0, 18, len(dt))
        users = base_users * (1 + trend + hour_season + weekend_boost + month_season + 0.35*event_boost) + noise_u
        users = np.clip(users, 10, None)

        # traffic dépend users + bruit + capacité locale
        capacity_factor = {"urban": 1.25, "suburban": 1.0, "rural": 0.85}[zone]
        noise_t = rng.normal(0, 35, len(dt))
        traffic = (base_traffic * capacity_factor) * (1 + trend + 0.9*hour_season + 0.6*weekend_boost + 0.7*month_season + 1.0*event_boost) \
                 + 0.45*(users - users.mean()) + noise_t
        traffic = np.clip(traffic, 1, None)

        df_cell = pd.DataFrame({
            "timestamp": dt,
            "cell_id": cell,
            "region": region,
            "zone_type": zone,
            "event_flag": event_flag.astype(int),
            "users": users.round(0).astype(int),
            "traffic_mbps": traffic.round(2),
        })
        rows.append(df_cell)

    return pd.concat(rows, ignore_index=True)
