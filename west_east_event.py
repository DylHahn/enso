from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================================================
# Global paper-style formatting
# ============================================================
mpl.rcParams.update({
    "font.family": "Arial",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "figure.titlesize": 22,
    "figure.titleweight": "bold",
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.2,
})

DATA_DIR = Path("data")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

EAST = dict(
    depth=5,
    lat0=0.0,
    lon0=215.0,
    half_lat=5.0,
    half_lon=25.0,
)

WEST_TEMPLATE = dict(
    lat0=5.0,
    lon0=150.0,
    half_lat=5.0,
    half_lon=10.0,
)

WEST_DEPTHS = [55, 85, 175, 205]
EVENT_YEARS = ["1983", "1992", "1998", "2016"]
PEAK_MONTHS = (11, 12, 1, 2, 3, 4, 5)
LEAD_MAX_MONTHS = 18
OUT_NAME = "fig_3.png"


def load_da(depth_m):
    ds = xr.open_dataset(DATA_DIR / f"godasClimatologyData_{depth_m}m.nc")
    try:
        da = ds["deepTemp"].load()
    finally:
        ds.close()
    return da


def lat_weights(lat):
    return xr.DataArray(
        np.cos(np.deg2rad(lat.values)),
        coords={"lat": lat},
        dims=("lat",),
    )


def box_anom_series(depth, lat0, lon0, half_lat, half_lon):
    da = load_da(depth)

    sub = da.sel(
        lat=slice(lat0 - half_lat, lat0 + half_lat),
        lon=slice(lon0 - half_lon, lon0 + half_lon),
    )

    clim = sub.groupby("time.month").mean("time")
    anom = sub.groupby("time.month") - clim

    w = lat_weights(sub["lat"])
    return anom.weighted(w).mean(("lat", "lon"))


def zscore(da):
    return (da - da.mean("time")) / da.std("time")


def find_east_peak_in_year_season(vals, times, year_str, peak_months, mode="max"):
    year = int(year_str)

    start = pd.to_datetime(f"{year - 1}-11-01")
    end = pd.to_datetime(f"{year}-05-31")

    times_pd = pd.to_datetime(times)
    mask = (
        (times_pd >= start)
        & (times_pd <= end)
        & (times_pd.month.isin(peak_months))
    )

    if not mask.any():
        return None, None

    idxs = np.nonzero(mask)[0]
    sub = vals[idxs]

    if mode == "max":
        i = idxs[np.nanargmax(sub)]
    else:
        i = idxs[np.nanargmin(sub)]

    return i, vals[i]


def find_west_min_around_east(W, time, i_east_idx, months_before=18, months_after=0):
    t_e = pd.to_datetime(time[i_east_idx])

    start = t_e - pd.DateOffset(months=months_before)
    end = t_e + pd.DateOffset(months=months_after)

    mask = (time >= start) & (time <= end)

    if not mask.any():
        return None, np.nan

    idxs = np.nonzero(mask)[0]
    i = idxs[np.nanargmin(W[idxs])]

    return i, W[i]


def make_multi_panel_og_logic(
    west_depths=WEST_DEPTHS,
    east_cfg=EAST,
    years=EVENT_YEARS,
    lead_max_months=LEAD_MAX_MONTHS,
    out_name=OUT_NAME,
):
    east_z = zscore(box_anom_series(**east_cfg))

    fig, axs = plt.subplots(
        len(west_depths),
        1,
        figsize=(14, 3.1 * len(west_depths)),
        sharex=True,
    )

    if len(west_depths) == 1:
        axs = [axs]

    fig.suptitle(
        "Western Pacific subsurface anomalies and Niño 3.4 surface response",
        fontsize=22,
        fontweight="bold",
        y=0.965,
    )

    handles = [
        plt.Line2D([0], [0], color="tab:blue", lw=2.2),
        plt.Line2D([0], [0], color="tab:red", lw=2.2),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="red", markeredgecolor="white", markersize=9),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="blue", markeredgecolor="white", markersize=9),
    ]

    labels = ["Western Pacific subsurface", "Niño 3.4 surface", "Eastern peak", "Western minimum"]

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
        fontsize=13,
    )

    panel_labels = [f"({chr(97 + i)})" for i in range(len(west_depths))]
    all_matches = []

    for p, west_depth in enumerate(west_depths):
        west_cfg = dict(depth=west_depth, **WEST_TEMPLATE)

        west_z = zscore(box_anom_series(**west_cfg))
        west_z_aligned, east_z_aligned = xr.align(west_z, east_z, join="inner")

        time = pd.to_datetime(west_z_aligned["time"].values)
        W = west_z_aligned.values.astype(float)
        E = east_z_aligned.values.astype(float)

        ax = axs[p]

        ax.plot(time, W, color="tab:blue", linewidth=1.8)
        ax.plot(time, E, color="tab:red", linewidth=1.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0)

        for y in years:
            i_e, east_val = find_east_peak_in_year_season(
                E,
                time,
                y,
                peak_months=PEAK_MONTHS,
                mode="max",
            )

            if i_e is None:
                continue

            months_after = 12 if west_depth >= 200 else 0

            i_w, west_val = find_west_min_around_east(
                W,
                time,
                i_east_idx=i_e,
                months_before=lead_max_months,
                months_after=months_after,
            )

            if i_w is None:
                continue

            east_date = time[i_e]
            west_date = time[i_w]

            all_matches.append(
                dict(
                    west_depth=west_depth,
                    event_year=y,
                    east_date=east_date,
                    east_val=east_val,
                    west_date=west_date,
                    west_val=west_val,
                    lead_months=(pd.Period(east_date, "M") - pd.Period(west_date, "M")).n,
                )
            )

            ax.scatter(
                east_date,
                east_val,
                s=105,
                facecolor="red",
                edgecolor="white",
                linewidth=1.0,
                zorder=6,
            )

            ax.annotate(
                pd.Timestamp(east_date).strftime("%Y-%m"),
                (east_date, east_val),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="red",
            )

            ax.scatter(
                west_date,
                west_val,
                s=105,
                facecolor="blue",
                edgecolor="white",
                linewidth=1.0,
                zorder=6,
            )

            ax.annotate(
                pd.Timestamp(west_date).strftime("%Y-%m"),
                (west_date, west_val),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="blue",
            )

        ax.set_title(
            f"Western Pacific {west_depth} m vs. Niño 3.4 surface",
            fontsize=15,
            fontweight="bold",
            pad=6,
        )

        ax.set_ylabel("Standardized anomaly (z)", fontsize=13)
        ax.grid(alpha=0.25, linestyle=":")

        ax.tick_params(axis="both", labelsize=12, width=1.1, length=5)

        ax.text(
            -0.055,
            1.03,
            panel_labels[p],
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=17,
            fontweight="bold",
        )

    axs[-1].set_xlabel("Year", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.91])

    out_path = FIG_DIR / out_name
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {out_path}")
    print("\nMatches: East peak -> West minimum")

    for m in all_matches:
        east_str = pd.Timestamp(m["east_date"]).strftime("%Y-%m")
        west_str = pd.Timestamp(m["west_date"]).strftime("%Y-%m")
        print(
            f"Depth {m['west_depth']:>3} m | "
            f"East {east_str} z={m['east_val']:.2f} -> "
            f"West {west_str} z={m['west_val']:.2f} | "
            f"lead={m['lead_months']} months"
        )

    return all_matches


def plot_lead_time_by_depth(all_matches, event_years):
    df = pd.DataFrame(all_matches)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(event_years)))
    event_color = {yr: col for yr, col in zip(event_years, colors)}

    for yr in event_years:
        sub = df[df["event_year"] == yr].sort_values("west_depth")
        if sub.empty:
            continue

        ax.plot(
            sub["lead_months"],
            sub["west_depth"],
            "o-",
            color=event_color[yr],
            linewidth=2.2,
            markersize=8,
            label=f"{yr} event",
        )

    mean_lead = df.groupby("west_depth")["lead_months"].mean().reset_index()

    ax.plot(
        mean_lead["lead_months"],
        mean_lead["west_depth"],
        "k--",
        linewidth=2.6,
        label="Mean across events",
        zorder=5,
    )

    ax.axhline(y=55, color="steelblue", linestyle=":", linewidth=2.3, alpha=0.8)
    ax.invert_yaxis()

    ax.set_xlabel("Forecast Lead Time (months)", fontsize=14)
    ax.set_ylabel("West Predictor Depth [m]", fontsize=14)
    ax.set_title("Lead Time by Depth", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    ax2 = axes[1]

    for yr in event_years:
        sub = df[df["event_year"] == yr].sort_values("west_depth")
        if sub.empty:
            continue

        ax2.plot(
            sub["west_val"].abs(),
            sub["west_depth"],
            "s-",
            color=event_color[yr],
            linewidth=2.2,
            markersize=8,
            label=f"{yr} event",
        )

    mean_amp = (
        df.groupby("west_depth")["west_val"]
        .apply(lambda x: x.abs().mean())
        .reset_index()
    )
    mean_amp.columns = ["west_depth", "mean_amp"]

    ax2.plot(
        mean_amp["mean_amp"],
        mean_amp["west_depth"],
        "k--",
        linewidth=2.6,
        label="Mean amplitude",
        zorder=5,
    )

    ax2.axhline(y=55, color="steelblue", linestyle=":", linewidth=2.3, alpha=0.8)
    ax2.invert_yaxis()

    ax2.set_xlabel("Precursor Amplitude |z|", fontsize=14)
    ax2.set_ylabel("West Predictor Depth [m]", fontsize=14)
    ax2.set_title("Precursor Signal Amplitude by Depth", fontsize=16, fontweight="bold")
    ax2.legend(fontsize=11, frameon=False)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)

    fig.suptitle(
        "Western Pacific Precursor Timing and Amplitude by Depth",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    out = FIG_DIR / "lead_time_amplitude_by_depth.png"
    plt.savefig(out, dpi=400, bbox_inches="tight")
    plt.show()

    print(f"Saved: {out}")


def make_summary_table(all_matches):
    df = pd.DataFrame(all_matches)

    lead_mean = (
        df.groupby("west_depth")["lead_months"]
        .mean()
        .round(2)
    )

    amp_mean = (
        df.groupby("west_depth")["west_val"]
        .apply(lambda x: np.abs(x).mean())
        .round(2)
    )

    table = pd.DataFrame({
        "Mean Lead Time (months)": lead_mean,
        "Mean Precursor Amplitude |z|": amp_mean,
    }).sort_index()

    print("\n" + "=" * 70)
    print("Western Pacific Precursor Summary by Depth")
    print("=" * 70)
    print(table.to_string())
    print("=" * 70)

    return table


if __name__ == "__main__":

    # Full-depth run for the summary table and lead/amplitude figure
    all_matches_full = make_multi_panel_og_logic(
        west_depths=np.arange(5, 215, 10),
        years=EVENT_YEARS,
        lead_max_months=18,
        out_name="fig_3_all_depths.png",
    )

    plot_lead_time_by_depth(
        all_matches_full,
        EVENT_YEARS,
    )

    table = make_summary_table(all_matches_full)

    # Final paper figure with selected depths only
    make_multi_panel_og_logic(
        west_depths=WEST_DEPTHS,
        years=EVENT_YEARS,
        lead_max_months=18,
        out_name="fig_3.png",
    )

