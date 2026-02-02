import numpy as np

def estimate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return y_true - y_pred

def saturation_probability(
    y_pred_point: float,
    residuals: np.ndarray,
    threshold: float,
    n_samples: int = 2000,
    seed: int = 42
) -> float:
    """
    Approxime P(y > threshold) où y = y_pred_point + epsilon,
    et epsilon est tiré des résidus observés (bootstrap).
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size == 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    eps = rng.choice(residuals, size=n_samples, replace=True)
    samples = y_pred_point + eps
    return float(np.mean(samples > threshold))

def risk_level(p: float) -> str:
    if p != p:  # NaN
        return "UNKNOWN"
    if p >= 0.6:
        return "HIGH"
    if p >= 0.25:
        return "MEDIUM"
    return "LOW"
