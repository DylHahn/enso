"""
Run jackknife sensitivity tests for event-based depth selection.

This script recomputes western/eastern event matches while leaving out one
El Nino event at a time. It is intended to test whether the selected western
Pacific precursor depth is stable across individual events.

Example
-------
python enso_jackknife.py

Outputs
-------
- stats/event_depth_all_events.csv
- stats/event_depth_jackknife_summary.csv
- stats/event_depth_jackknife_wide.csv
"""


import os
import numpy as np
import pandas as pd
import xarray as xr

from west_east_event import (
    EAST,
    WEST_TEMPLATE,
    EVENT_YEARS,
    PEAK_MONTHS,
    LEAD_MAX_MONTHS,
    box_anom_series,
    zscore,
    find_east_peak_in_year_season,
    find_west_min_around_east,
)


def compute_event_matches(
    west_depths=np.arange(5, 215, 10),
    event_years=EVENT_YEARS,
    months_before=LEAD_MAX_MONTHS,
    months_after=0,
):
    """
    Compute event matches without plotting.

    For each western depth and event, identify the eastern peak and the
    most negative western anomaly within the pre-peak window.
    """
    east_z = zscore(box_anom_series(**EAST))
    records = []

    for west_depth in west_depths:
        west_cfg = dict(depth=int(west_depth), **WEST_TEMPLATE)
        west_z = zscore(box_anom_series(**west_cfg))
        west_z_aligned, east_z_aligned = xr.align(west_z, east_z, join="inner")

        time = pd.to_datetime(west_z_aligned["time"].values)
        W = west_z_aligned.values.astype(float)
        E = east_z_aligned.values.astype(float)

        for event_year in event_years:
            i_e, east_val = find_east_peak_in_year_season(
                E,
                time,
                event_year,
                peak_months=PEAK_MONTHS,
                mode="max",
            )
            if i_e is None:
                continue

            i_w, west_val = find_west_min_around_east(
                W,
                time,
                i_east_idx=i_e,
                months_before=months_before,
                months_after=months_after,
            )
            if i_w is None:
                continue

            east_date = pd.Timestamp(time[i_e])
            west_date = pd.Timestamp(time[i_w])
            lead_months = (pd.Period(east_date, "M") - pd.Period(west_date, "M")).n
            records.append({
                "event_year": str(event_year),
                "west_depth_m": int(west_depth),
                "east_date": east_date.strftime("%Y-%m"),
                "west_date": west_date.strftime("%Y-%m"),
                "east_z": float(east_val),
                "west_z": float(west_val),
                "west_abs_z": float(abs(west_val)),
                "lead_months": int(lead_months),
            })

    return pd.DataFrame(records)


def summarize_depth_selection(df, event_subset, min_mean_lead=0.0):
    """
    For a subset of events, compute mean amplitude and lead by depth.

    The selected depth is the depth with the largest mean |z| among depths
    with mean lead time >= min_mean_lead. This matches the paper's logic:
    strong standardized event amplitude while retaining positive lead time.
    """
    sub = df[df["event_year"].isin([str(e) for e in event_subset])].copy()

    summary = (
        sub.groupby("west_depth_m")
        .agg(
            n_events=("event_year", "count"),
            mean_lead_months=("lead_months", "mean"),
            mean_abs_z=("west_abs_z", "mean"),
            median_abs_z=("west_abs_z", "median"),
        )
        .reset_index()
    )

    eligible = summary[summary["mean_lead_months"] >= min_mean_lead].copy()
    if eligible.empty:
        selected = summary.loc[summary["mean_abs_z"].idxmax()]
    else:
        selected = eligible.loc[eligible["mean_abs_z"].idxmax()]

    return summary, selected


def run_event_depth_jackknife(
    west_depths=np.arange(5, 215, 10),
    event_years=EVENT_YEARS,
    months_before=LEAD_MAX_MONTHS,
    months_after=0,
    min_mean_lead=0.0,
    out_prefix="stats/event_depth_jackknife",
):
    """Run full-sample and leave-one-event-out depth-selection tests."""
    os.makedirs("stats", exist_ok=True)

    df = compute_event_matches(
        west_depths=west_depths,
        event_years=event_years,
        months_before=months_before,
        months_after=months_after,
    )

    all_events_path = "stats/event_depth_all_events.csv"
    df.round(4).to_csv(all_events_path, index=False)

    records = []
    all_events = [str(e) for e in event_years]

    # Full sample selection.
    full_summary, full_selected = summarize_depth_selection(
        df,
        event_subset=all_events,
        min_mean_lead=min_mean_lead,
    )
    records.append({
        "case": "all_events",
        "removed_event": "none",
        "events_used": ",".join(all_events),
        "selected_depth_m": int(full_selected["west_depth_m"]),
        "selected_mean_abs_z": float(full_selected["mean_abs_z"]),
        "selected_mean_lead_months": float(full_selected["mean_lead_months"]),
        "selected_n_events": int(full_selected["n_events"]),
    })

    # Leave-one-event-out selections.
    for removed in all_events:
        used = [e for e in all_events if e != removed]
        summary, selected = summarize_depth_selection(
            df,
            event_subset=used,
            min_mean_lead=min_mean_lead,
        )
        records.append({
            "case": "leave_one_event_out",
            "removed_event": removed,
            "events_used": ",".join(used),
            "selected_depth_m": int(selected["west_depth_m"]),
            "selected_mean_abs_z": float(selected["mean_abs_z"]),
            "selected_mean_lead_months": float(selected["mean_lead_months"]),
            "selected_n_events": int(selected["n_events"]),
        })

    jk = pd.DataFrame(records)
    jk_path = f"{out_prefix}_summary.csv"
    jk.round(4).to_csv(jk_path, index=False)

    # Wide depth-band summary: shows 45, 55, 65 m are a band.
    band_depths = [35, 45, 55, 65, 75, 85]
    band_rows = []
    for case_name, removed, used in [("all_events", "none", all_events)] + [
        ("leave_one_event_out", r, [e for e in all_events if e != r]) for r in all_events
    ]:
        summary, _ = summarize_depth_selection(df, used, min_mean_lead=min_mean_lead)
        row = {"case": case_name, "removed_event": removed}
        for d in band_depths:
            drow = summary[summary["west_depth_m"] == d]
            if drow.empty:
                row[f"depth_{d}m_mean_abs_z"] = np.nan
                row[f"depth_{d}m_mean_lead"] = np.nan
            else:
                row[f"depth_{d}m_mean_abs_z"] = float(drow.iloc[0]["mean_abs_z"])
                row[f"depth_{d}m_mean_lead"] = float(drow.iloc[0]["mean_lead_months"])
        band_rows.append(row)

    band_df = pd.DataFrame(band_rows)
    band_path = f"{out_prefix}_wide.csv"
    band_df.round(4).to_csv(band_path, index=False)

    print("\nEvent-depth jackknife sensitivity")
    print("---------------------------------")
    print(jk.round(3).to_string(index=False))
    print(f"\nSaved: {all_events_path}")
    print(f"Saved: {jk_path}")
    print(f"Saved: {band_path}")

    return df, jk, band_df


if __name__ == "__main__":
    run_event_depth_jackknife(
        west_depths=np.arange(5, 215, 10),
        event_years=EVENT_YEARS,
        months_before=LEAD_MAX_MONTHS,
        months_after=0,
        min_mean_lead=0.0,
    )
