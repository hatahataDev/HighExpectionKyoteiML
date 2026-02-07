import numpy as np
import pandas as pd

try:
    from sklearn.metrics import mean_squared_error
    try:
        from sklearn.metrics import root_mean_squared_error
    except ImportError:
        root_mean_squared_error = None
except ImportError as exc:
    raise ImportError("scikit-learn is required. Install with: pip install scikit-learn") from exc


def convert_racetime_to_seconds(rt_series: pd.Series) -> pd.Series:
    rt = rt_series.astype(float)
    minutes = np.floor(rt / 1000)
    seconds = (rt % 1000) / 10
    return minutes * 60 + seconds


def compute_rmse(y_true, y_pred) -> float:
    """Compute RMSE across scikit-learn versions."""
    if root_mean_squared_error is not None:
        return float(root_mean_squared_error(y_true, y_pred))
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
