"""
Estimate block-bootstrap uncertainty for baseline ENSO comparisons.

The bootstrap resamples contiguous validation-month blocks to approximate
serial dependence in monthly ENSO anomalies. Outputs are written to the
``stats/`` directory.

Example
-------
python enso_bootstrap.py

Outputs
-------
- stats/bootstrap_long.csv
- stats/bootstrap_wide.csv
"""


import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from enso import EnsoLinearModel


def safe_corr(y_true, y_pred):
    """Return a finite-sample Pearson correlation with safeguards for invalid inputs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) < 3:
        return np.nan
    if np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def rmse(y_true, y_pred):
    """Return the root-mean-square error after removing invalid pairs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def block_bootstrap_metric_ci(
    y_true,
    y_pred,
    metric_func,
    block_len=12,
    n_boot=2000,
    ci=95,
    random_seed=42,
):
    """
    Bootstrap confidence interval using moving blocks of validation months.

    This resamples contiguous blocks with replacement until the resampled
    series has the same length as the original validation series.
    """
    rng = np.random.default_rng(random_seed)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    n = len(y_true)
    if n < block_len + 2:
        return np.nan, np.nan

    starts = np.arange(0, n - block_len + 1)
    values = []

    for _ in range(n_boot):
        sample_idx = []
        while len(sample_idx) < n:
            s = int(rng.choice(starts))
            sample_idx.extend(range(s, s + block_len))
        sample_idx = np.asarray(sample_idx[:n], dtype=int)
        values.append(metric_func(y_true[sample_idx], y_pred[sample_idx]))

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan

    alpha = (100 - ci) / 2
    return (
        float(np.percentile(values, alpha)),
        float(np.percentile(values, 100 - alpha)),
    )


def build_validation_predictions(
    obj,
    target_mode,
    lead,
    predictor_depth=55,
    predictand_depth=5,
    predictor_lat_range=(0, 10),
    predictor_lon_range=(140, 160),
    target_lat_range=(-5, 5),
    target_lon_range=(190, 240),
):
    """
    Return y_true, 55 m model prediction, persistence prediction,
    and zero-anomaly climatology prediction for one lead and target.
    """
    x_series = obj.build_boxmean_series(
        depth=predictor_depth,
        lat_range=predictor_lat_range,
        lon_range=predictor_lon_range,
    )

    if target_mode == "benchmark":
        y_series = obj.load_target_series(predictand_depth)
    elif target_mode == "boxmean":
        y_series = obj.build_boxmean_series(
            depth=predictand_depth,
            lat_range=target_lat_range,
            lon_range=target_lon_range,
        )
    else:
        raise ValueError("target_mode must be 'benchmark' or 'boxmean'.")

    X_train, y_train = obj.assemble_boxmean_xy(
        x_series, y_series, obj.start_stop_list_train, lead
    )
    X_val, y_val_future = obj.assemble_boxmean_xy(
        x_series, y_series, obj.start_stop_list_val, lead
    )

    # Build persistence baseline: y_hat(t+h) = y(t)
    val_start = pd.to_datetime(obj.start_stop_list_val[0])
    val_end = pd.to_datetime(obj.start_stop_list_val[1])
    init_dates = pd.date_range(val_start, val_end, freq="MS")
    future_dates = init_dates + pd.DateOffset(months=int(lead))

    valid_dates = future_dates.isin(y_series.index) & init_dates.isin(y_series.index)
    init_dates = init_dates[valid_dates]
    future_dates = future_dates[valid_dates]

    y_init = y_series.loc[init_dates]
    y_true = y_series.loc[future_dates]

    n = min(len(X_val), len(y_true), len(y_init))
    X_val = X_val[:n]
    y_true = pd.Series(y_true.values[:n], index=future_dates[:n])
    y_init = pd.Series(y_init.values[:n], index=future_dates[:n])

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_model = pd.Series(model.predict(X_val), index=y_true.index)

    y_persist = y_init.copy()
    y_clim = pd.Series(np.zeros(len(y_true)), index=y_true.index)

    return y_true, {
        "west_55m_linear": y_model,
        "persistence": y_persist,
        "climatology_zero": y_clim,
    }


def run_bootstrap(
    lead_times=(1, 3, 6, 9),
    block_len=12,
    n_boot=2000,
    ci=95,
    random_seed=42,
    out_prefix="stats/bootstrap",
):
    """Run the block-bootstrap workflow and save long and wide CSV summaries."""
    os.makedirs("stats", exist_ok=True)
    obj = EnsoLinearModel()
    records = []

    target_label = {
        "benchmark": "Benchmark Nino 3.4",
        "boxmean": "GODAS surface box",
    }

    for target_mode in ("benchmark", "boxmean"):
        for lead in lead_times:
            y_true, preds = build_validation_predictions(obj, target_mode, lead)

            for model_name, y_pred in preds.items():
                corr = safe_corr(y_true.values, y_pred.values)
                this_rmse = rmse(y_true.values, y_pred.values)

                corr_lo, corr_hi = block_bootstrap_metric_ci(
                    y_true.values,
                    y_pred.values,
                    safe_corr,
                    block_len=block_len,
                    n_boot=n_boot,
                    ci=ci,
                    random_seed=random_seed + 10 * lead,
                )
                rmse_lo, rmse_hi = block_bootstrap_metric_ci(
                    y_true.values,
                    y_pred.values,
                    rmse,
                    block_len=block_len,
                    n_boot=n_boot,
                    ci=ci,
                    random_seed=random_seed + 100 * lead,
                )

                records.append({
                    "target_mode": target_mode,
                    "target_label": target_label[target_mode],
                    "lead_months": lead,
                    "model": model_name,
                    "n_val": len(y_true),
                    "corr": corr,
                    "corr_ci_low": corr_lo,
                    "corr_ci_high": corr_hi,
                    "rmse": this_rmse,
                    "rmse_ci_low": rmse_lo,
                    "rmse_ci_high": rmse_hi,
                    "block_len_months": block_len,
                    "n_boot": n_boot,
                })

    long_df = pd.DataFrame(records)
    long_path = f"{out_prefix}_long.csv"
    long_df.round(4).to_csv(long_path, index=False)

    # Compact wide table for manuscript use.
    rows = []
    for target_mode in ("benchmark", "boxmean"):
        for lead in lead_times:
            sub = long_df[(long_df["target_mode"] == target_mode) & (long_df["lead_months"] == lead)]
            row = {
                "target_label": target_label[target_mode],
                "lead_months": lead,
            }
            for model_name in ("west_55m_linear", "persistence", "climatology_zero"):
                m = sub[sub["model"] == model_name].iloc[0]
                prefix = model_name.replace("west_55m_linear", "west55m")
                row[f"{prefix}_corr"] = m["corr"]
                row[f"{prefix}_corr_ci"] = f"[{m['corr_ci_low']:.2f}, {m['corr_ci_high']:.2f}]"
                row[f"{prefix}_rmse"] = m["rmse"]
                row[f"{prefix}_rmse_ci"] = f"[{m['rmse_ci_low']:.2f}, {m['rmse_ci_high']:.2f}]"
            rows.append(row)

    wide_df = pd.DataFrame(rows)
    wide_path = f"{out_prefix}_wide.csv"
    wide_df.to_csv(wide_path, index=False)

    print("\nBlock-bootstrap uncertainty estimates")
    print("------------------------------------------------")
    print(wide_df.to_string(index=False))
    print(f"\nSaved: {long_path}")
    print(f"Saved: {wide_path}")

    return long_df, wide_df


if __name__ == "__main__":
    run_bootstrap(
        lead_times=(1, 3, 6, 9),
        block_len=12,
        n_boot=2000,
        ci=95,
        random_seed=42,
    )
