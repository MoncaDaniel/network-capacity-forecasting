from dataclasses import dataclass

@dataclass(frozen=True)
class ForecastConfig:
    horizon_days_short: int = 7
    horizon_days_long: int = 30
    freq: str = "h"

    # Seuils "capacité" par type de zone (MVP réaliste)
    saturation_threshold_by_zone: dict = None

    def __post_init__(self):
        if self.saturation_threshold_by_zone is None:
            object.__setattr__(self, "saturation_threshold_by_zone", {
                "urban": 800.0,
                "suburban": 450.0,
                "rural": 250.0,
            })
