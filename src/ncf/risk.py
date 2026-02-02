import numpy as np

def estimate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return y_true - y_pred

def _calibrate_residuals(
    residuals: np.ndarray,
    method: str = "winsor",
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    clip_sigma: float = 3.0,
) -> np.ndarray:
    """
    Calibration robuste des résidus pour éviter des queues extrêmes
    (souvent responsables de p_saturation irréalistes).

    - winsor: clamp entre quantiles [lower_q, upper_q]
    - sigma: clamp entre [-clip_sigma*std, +clip_sigma*std] (après centrage)
    - both: winsor puis sigma
    """
    r = np.asarray(residuals, dtype=float)
    if r.size == 0:
        return r

    # centrer pour que la calibration soit stable
    r = r - np.nanmean(r)

    if method in ("winsor", "both"):
        lo, hi = np.nanquantile(r, [lower_q, upper_q])
        r = np.clip(r, lo, hi)

    if method in ("sigma", "both"):
        std = np.nanstd(r)
        if std > 0:
            bound = clip_sigma * std
            r = np.clip(r, -bound, bound)

    return r

def saturation_probability(
    y_pred_point: float,
    residuals: np.ndarray,
    threshold: float,
    n_samples: int = 2000,
    seed: int = 42,
    calibrate: str = "both",
) -> float:
    """
    Approxime P(y > threshold) où y = y_pred_point + epsilon
    et epsilon est tiré des résidus observés (bootstrap), calibrés.
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size == 0:
        return float("nan")

    r = _calibrate_residuals(residuals, method=calibrate)

    rng = np.random.default_rng(seed)
    eps = rng.choice(r, size=n_samples, replace=True)
    samples = y_pred_point + eps
    return float(np.mean(samples > threshold))

def window_saturation_probability(
    y_pred_series: np.ndarray,
    residuals: np.ndarray,
    threshold: float,
    n_paths: int = 2000,
    seed: int = 42,
    calibrate: str = "both",
) -> float:
    """
    Approxime P(max(y_t) > threshold) sur une fenêtre,
    avec y_t = y_pred_t + eps_t, eps_t tirés des résidus (bootstrap i.i.d),
    après calibration robuste des résidus.
    """
    y_pred_series = np.asarray(y_pred_series, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size == 0 or y_pred_series.size == 0:
        return float("nan")

    r = _calibrate_residuals(residuals, method=calibrate)

    rng = np.random.default_rng(seed)
    eps = rng.choice(r, size=(n_paths, y_pred_series.size), replace=True)
    sims = y_pred_series.reshape(1, -1) + eps
    return float(np.mean(np.max(sims, axis=1) > threshold))

def risk_level(p: float) -> str:
    if p != p:  # NaN
        return "UNKNOWN"
    if p >= 0.6:
        return "HIGH"
    if p >= 0.25:
        return "MEDIUM"
    return "LOW"
