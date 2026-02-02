from dataclasses import dataclass

@dataclass(frozen=True)
class ForecastConfig:
    horizon_days_short: int = 7
    horizon_days_long: int = 30
    freq: str = "h"  # hourly
    saturation_threshold_mbps: float = 800.0  # exemple, à ajuster
