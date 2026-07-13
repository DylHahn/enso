"""
Generate the main ENSO analysis figures and regression diagnostics.

This module supports the manuscript on western tropical Pacific subsurface
temperature anomalies as ENSO precursors. It loads GODAS monthly potential
temperature files from ``data/``, builds anomaly fields and box-mean time
series, evaluates ordinary least-squares forecast experiments, and saves
figures and summary statistics to ``figures/`` and ``stats/``.

Expected inputs
---------------
- data/godasClimatologyData_<depth>m.nc
- data/movingAverageAnomalies<depth>m.txt

Typical use
-----------
Run selected blocks from the ``__main__`` section after placing the required
input files in ``data/``.
"""


import os
import time

import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Global style
mpl.rcParams.update({
    "font.family": "Arial",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",

    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,

    "figure.titlesize": 22,
    "figure.titleweight": "bold",

    "lines.linewidth": 2.4,
    "axes.linewidth": 1.4,
})

pio.renderers.default = "browser"
DEG = "\N{DEGREE SIGN}"


def lon_label(x, _):
    """Return a longitude label using east/west notation."""
    if np.isnan(x):
        return ""
    x = float(x)
    if x <= 180:
        return f"{int(x)}{DEG}E"
    w = int(360 - x)
    return f"{w}{DEG}W" if w != 0 else f"180{DEG}"


def lat_label(y, _):
    """Return a latitude label using north/south notation."""
    if np.isnan(y):
        return ""
    y = int(round(y))
    if y > 0:
        return f"{y}{DEG}N"
    if y < 0:
        return f"{-y}{DEG}S"
    return f"0{DEG}"

def lon_formatter(x, pos=None):
    """Format longitude ticks for matplotlib axes."""
    if x <= 180:
        return f"{int(x)}°E"
    else:
        return f"{int(360 - x)}°W"

def lat_formatter(y, pos=None):
    """Format latitude ticks for matplotlib axes."""
    if y > 0:
        return f"{int(y)}°N"
    elif y < 0:
        return f"{int(-y)}°S"
    else:
        return "0°"
def sel_time_nearest(da, t):
    """Select the nearest available time from an xarray object."""
    return da.sel(time=np.datetime64(pd.Timestamp(t)), method="nearest")


def to_anomaly(da, clim_start="1980-01-01", clim_end="2023-12-31"):
    """Convert monthly fields to anomalies relative to a climatological baseline."""
    base = da.sel(time=slice(clim_start, clim_end))
    clim = base.groupby("time.month").mean("time")
    anom = da.groupby("time.month") - clim
    anom.attrs["units"] = "degC"
    return anom


def _ensure_output_dirs():
    """Create output directories used by the analysis scripts."""
    for path in ("stats", "figures"):
        os.makedirs(path, exist_ok=True)


def _plot_monthly_maps_2x3(
    ds_path,
    title,
    months,
    titles,
    rect_lat_range,
    rect_lon_range,
    out_name,
    vmin,
    vmax,
    rect_edgecolor,
    rect_text_color="white",
    rect_text_path_effects=None,
):
    """Plot a two-column, three-row set of monthly anomaly maps."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker
    import matplotlib.patheffects as pe

    if rect_text_path_effects is None:
        rect_text_path_effects = [pe.withStroke(linewidth=3, foreground="#1f2937")]

    mpl.rcParams.update({
        "font.family": "Arial",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    map_crs = ccrs.PlateCarree(central_longitude=180)
    data_crs = ccrs.PlateCarree()

    ds = xr.open_dataset(ds_path)
    try:
        anom = to_anomaly(ds["deepTemp"])

        fig = plt.figure(figsize=(13, 10.5))

        gs = fig.add_gridspec(
            nrows=3,
            ncols=3,
            width_ratios=[1, 1, 0.035],
            wspace=0.20,
            hspace=0.01,
        )

        axs = []
        for i in range(6):
            row = i // 2
            col = i % 2
            axs.append(fig.add_subplot(gs[row, col], projection=map_crs))

        fig.suptitle(title, fontsize=26, y=0.98)

        cmap = plt.get_cmap("jet").copy()
        cmap.set_bad((1, 1, 1, 0))

        xticks = [120, 150, 180, 210, 240, 270]
        yticks = [-20, -10, 0, 10, 20]

        mappable = None

        for i, month in enumerate(months):
            temp = sel_time_nearest(anom, month).squeeze()
            temp = temp.sel(lat=slice(-20, 20), lon=slice(120, 280))
            temp = temp.where(np.isfinite(temp))

            ax = axs[i]
            ax.set_extent([120, 280, -20, 20], crs=data_crs)

            pcm = ax.pcolormesh(
                temp["lon"],
                temp["lat"],
                temp,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="auto",
                transform=data_crs,
                zorder=1,
            )
            mappable = pcm

            ax.add_feature(
                cfeature.LAND,
                facecolor="white",
                edgecolor="black",
                linewidth=0.45,
                zorder=3,
            )
            ax.coastlines(resolution="50m", linewidth=0.65, zorder=4)

            # Gridlines only, no Cartopy auto-labels
            ax.gridlines(
                crs=data_crs,
                linewidth=0.4,
                linestyle=":",
                alpha=0.4,
                draw_labels=False,
                zorder=2,
            )

            # Manual ticks: this works better with central_longitude=180
            xticks = [120, 150, 180, 210, 240, 270]
            xlabels = ["120°E", "150°E", "180°", "150°W", "120°W", "90°W"]

            yticks = [-20, -10, 0, 10, 20]
            ylabels = ["20°S", "10°S", "0°", "10°N", "20°N"]

            ax.set_xticks(xticks, crs=data_crs)
            ax.set_xticklabels(xlabels, fontsize=12, fontweight="bold")

            ax.set_yticks(yticks, crs=data_crs)
            ax.set_yticklabels(ylabels, fontsize=12, fontweight="bold")

            ax.tick_params(
                axis="both",
                which="major",
                direction="out",
                length=5,
                width=1.1,
                labelsize=12,
            )

            label = f"({chr(97 + i)}) {titles[i]}"

            ax.set_title(
                f"({chr(97 + i)}) {titles[i]}",
                fontsize=13,
                fontweight="bold",
                pad=6
            )

            rect = patches.Rectangle(
                (rect_lon_range[0], rect_lat_range[0]),
                rect_lon_range[1] - rect_lon_range[0],
                rect_lat_range[1] - rect_lat_range[0],
                linewidth=2.0,
                edgecolor=rect_edgecolor,
                facecolor="none",
                transform=data_crs,
                zorder=7,
            )
            ax.add_patch(rect)

        cax = fig.add_subplot(gs[:, 2])
        cbar = fig.colorbar(mappable, cax=cax, orientation="vertical", extend="both")
        cbar.ax.set_title("°C", fontsize=16, pad=6)
        cbar.ax.tick_params(labelsize=13, width=1.2, length=6)

        fig.subplots_adjust(
            top=0.9,
            left=0.07,
            right=0.92,
            bottom=0.04,
        )

        plt.savefig(os.path.join("figures", out_name), dpi=400, bbox_inches="tight")
        plt.show()

    finally:
        ds.close()


def generate_figure2_like_paper():
    """Generate the western/eastern Pacific event-map figure."""
    months = [
        "1998-01-16",
        "1998-02-16",
        "1998-03-16",
        "1998-04-16",
        "1998-05-16",
        "1998-06-16",
    ]
    titles = [
        "January 1998",
        "February 1998",
        "March 1998",
        "April 1998",
        "May 1998",
        "June 1998",
    ]
    _plot_monthly_maps_2x3(
        ds_path="data/godasClimatologyData_5m.nc",
        title="Predictors and predictands at 5-meter depth ",
        months=months,
        titles=titles,
        rect_lat_range=(-10, 0),
        rect_lon_range=(195, 225),
        out_name="figure2_surface_stack.png",
        vmin=-5,
        vmax=5,
        rect_edgecolor="red",
        rect_text_color="white",
        rect_text_path_effects=[pe.withStroke(linewidth=3, foreground="#1f2937")],
        # xlabel_size=15,
    )


def generate_figure3_like_paper():

    """Generate the selected-depth precursor time-series figure."""
    months = [
        "1998-01-16",
        "1998-02-16",
        "1998-03-16",
        "1998-04-16",
        "1998-05-16",
        "1998-06-16",
    ]
    titles = [
        "January 1998",
        "February 1998",
        "March 1998",
        "April 1998",
        "May 1998",
        "June 1998",
    ]
    _plot_monthly_maps_2x3(
        ds_path="data/godasClimatologyData_55m.nc",
        title="Predictors and predictands at 55-meter depth ",
        months=months,
        titles=titles,
        rect_lat_range=(0, 10),
        rect_lon_range=(135, 165),
        out_name="figure3_deep_stack.png",
        vmin=-8,
        vmax=8,
        rect_edgecolor="navy",
        rect_text_color="white",
        rect_text_path_effects=[pe.withStroke(linewidth=3, foreground="#1f2937")],
        # xlabel_size=14,
    )

def generate_deep_to_surface_event_maps():
    """Generate maps comparing subsurface and surface event anomalies."""
    import os
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.patches as patches
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    mpl.rcParams.update({
        "font.family": "Arial",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    # -------------------------------------------------
    # Event pairs from your time series
    # -------------------------------------------------
    event_pairs = [
        ("1982-11-16", "1983-01-16", "1982–83"),
        ("1991-12-16", "1992-02-16", "1991–92"),
        ("1997-12-16", "1997-12-16", "1997–98"),
        ("2015-07-16", "2015-12-16", "2015–16"),
    ]

    # -------------------------------------------------
    # Data paths
    # -------------------------------------------------
    west_depth = 55
    east_depth = 5

    west_path = os.path.join("data", f"godasClimatologyData_{west_depth}m.nc")
    east_path = os.path.join("data", f"godasClimatologyData_{east_depth}m.nc")

    # -------------------------------------------------
    # Boxes
    # -------------------------------------------------
    # West box: 0–10N, 140–160E
    west_lat_range = (0, 10)
    west_lon_range = (140, 160)

    # Niño 3.4 / East box: 5S–5N, 170W–120W
    # 170W = 190E, 120W = 240E
    east_lat_range = (-5, 5)
    east_lon_range = (190, 240)

    # -------------------------------------------------
    # Map settings
    # -------------------------------------------------
    lat_range = slice(-20, 20)
    lon_range = slice(120, 280)

    map_crs = ccrs.PlateCarree(central_longitude=180)
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(15, 13))

    gs = fig.add_gridspec(
        nrows=4,
        ncols=3,
        width_ratios=[1, 1, 0.045],
        wspace=0.20,
        hspace=0.28,
    )

    axs = []
    for r in range(4):
        row_axes = []
        for c in range(2):
            row_axes.append(fig.add_subplot(gs[r, c], projection=map_crs))
        axs.append(row_axes)

    fig.suptitle(
        "Western Pacific subsurface precursor and eastern Pacific surface response",
        fontsize=24,
        y=0.97,
        fontweight="bold",
    )

    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad((1, 1, 1, 0))

    xticks = [120, 150, 180, 210, 240, 270]
    xlabels = ["120°E", "150°E", "180°", "150°W", "120°W", "90°W"]

    yticks = [-20, -10, 0, 10, 20]
    ylabels = ["20°S", "10°S", "0°", "10°N", "20°N"]

    # Use separate color limits for deep and surface
    west_vmin, west_vmax = -8, 8
    east_vmin, east_vmax = -5, 5

    mappable = None

    west_ds = xr.open_dataset(west_path)
    east_ds = xr.open_dataset(east_path)

    try:
        west_anom = to_anomaly(west_ds["deepTemp"])
        east_anom = to_anomaly(east_ds["deepTemp"])

        for i, (west_date, east_date, event_label) in enumerate(event_pairs):

            # -------------------------------------------------
            # LEFT: deep west signal at 55 m
            # -------------------------------------------------
            ax = axs[i][0]

            west_temp = (
                sel_time_nearest(west_anom, west_date)
                .sel(lat=lat_range, lon=lon_range)
                .squeeze()
            )

            west_temp = west_temp.where(np.isfinite(west_temp))

            pcm = ax.pcolormesh(
                west_temp["lon"],
                west_temp["lat"],
                west_temp,
                cmap=cmap,
                vmin=west_vmin,
                vmax=west_vmax,
                shading="auto",
                transform=data_crs,
                zorder=1,
            )

            ax.add_patch(
                patches.Rectangle(
                    (west_lon_range[0], west_lat_range[0]),
                    west_lon_range[1] - west_lon_range[0],
                    west_lat_range[1] - west_lat_range[0],
                    linewidth=2.5,
                    edgecolor="purple",
                    facecolor="none",
                    transform=data_crs,
                    zorder=7,
                )
            )

            ax.set_title(
                f"({chr(97 + 2*i)}) Western Pacific 55 m, {pd.Timestamp(west_date).strftime('%B %Y')}",
                fontsize=14,
                pad=6,
            )

            # -------------------------------------------------
            # RIGHT: later surface east / Niño 3.4 signal
            # -------------------------------------------------
            ax = axs[i][1]

            east_temp = (
                sel_time_nearest(east_anom, east_date)
                .sel(lat=lat_range, lon=lon_range)
                .squeeze()
            )

            east_temp = east_temp.where(np.isfinite(east_temp))

            pcm = ax.pcolormesh(
                east_temp["lon"],
                east_temp["lat"],
                east_temp,
                cmap=cmap,
                vmin=east_vmin,
                vmax=east_vmax,
                shading="auto",
                transform=data_crs,
                zorder=1,
            )

            mappable = pcm

            ax.add_patch(
                patches.Rectangle(
                    (east_lon_range[0], east_lat_range[0]),
                    east_lon_range[1] - east_lon_range[0],
                    east_lat_range[1] - east_lat_range[0],
                    linewidth=2.5,
                    edgecolor="white",
                    facecolor="none",
                    transform=data_crs,
                    zorder=7,
                )
            )

            ax.set_title(
                f"({chr(98 + 2*i)}) Niño 3.4 surface, {pd.Timestamp(east_date).strftime('%B %Y')}",
                fontsize=14,
                pad=6,
            )

            # -------------------------------------------------
            # Style both panels in the row
            # -------------------------------------------------
            for ax in axs[i]:
                ax.set_extent([120, 280, -20, 20], crs=data_crs)

                ax.add_feature(
                    cfeature.LAND,
                    facecolor="black",
                    edgecolor="black",
                    linewidth=0.45,
                    zorder=3,
                )

                ax.coastlines(
                    resolution="50m",
                    linewidth=0.65,
                    zorder=4,
                )

                ax.gridlines(
                    crs=data_crs,
                    linewidth=0.4,
                    linestyle=":",
                    alpha=0.4,
                    draw_labels=False,
                    zorder=2,
                )

                ax.set_xticks(xticks, crs=data_crs)
                ax.set_xticklabels(xlabels, fontsize=11, fontweight="bold")

                ax.set_yticks(yticks, crs=data_crs)
                ax.set_yticklabels(ylabels, fontsize=11, fontweight="bold")

                ax.tick_params(
                    axis="both",
                    which="major",
                    direction="out",
                    length=5,
                    width=1.1,
                    labelsize=11,
                )

        # -------------------------------------------------
        # Shared colorbar
        # -------------------------------------------------
        cax = fig.add_subplot(gs[:, 2])
        cbar = fig.colorbar(
            mappable,
            cax=cax,
            orientation="vertical",
            extend="both",
        )

        cbar.ax.set_title("°C", fontsize=16, pad=6)
        cbar.ax.tick_params(labelsize=13, width=1.2, length=6)

        plt.savefig(
            os.path.join("figures", "fig_2.png"),
            dpi=400,
            bbox_inches="tight",
        )

        plt.show()

    finally:
        west_ds.close()
        east_ds.close()


class EnsoModel:
    """Base class for loading GODAS data and assembling ENSO predictor/target arrays."""
    def __init__(self):
        """Initialize dataset dates, train/validation windows, and output directories."""
        self.dataset_begin = "1980-01-01"
        self.dataset_end = "2023-12-31"
        self.start_stop_list_train = ["1980-01-01", "1995-12-31"]
        self.start_stop_list_val = ["1997-01-01", "2023-12-31"]
        self.save_time_series_plots = False
        _ensure_output_dirs()

    @staticmethod
    def obtain_dir(predictor_depth=5, predictand_depth=5):
        """Return the predictor NetCDF path and predictand text-file path."""
        return [
            os.path.join("data", f"godasClimatologyData_{predictor_depth}m.nc"),
            os.path.join("data", f"movingAverageAnomalies{predictand_depth}m.txt"),
        ]

    @staticmethod
    def corr_rmse_fcn(y_true, predictions):
        """Return Pearson correlation and RMSE for a prediction vector."""
        mse = mean_squared_error(y_true, predictions)
        rmse = np.sqrt(mse)
        return scipy.stats.pearsonr(y_true, predictions)[0], rmse


    def _load_target_txt_series(self, predictand_depth):
        """Load a monthly target anomaly series from the manuscript text-format input."""
        anomaly_path = self.obtain_dir(predictand_depth=predictand_depth)[1]
        vals = []

        with open(anomaly_path) as f:
            for line in f:
                vals.extend(map(float, line.split()[1:]))

        y = pd.Series(
            vals,
            index=pd.date_range(self.dataset_begin, freq="MS", periods=len(vals))
        )
        y.index = pd.to_datetime(y.index)

        return y.sort_index()


    def _load_predictor_anomaly_array(self, predictor_depth):
        """Load a predictor depth field and convert it to monthly anomalies."""
        godas_path = self.obtain_dir(predictor_depth=predictor_depth)[0]

        with xr.open_dataset(godas_path) as ds:
            deep_temp = ds["deepTemp"].sel(
                lat=slice(-20, 20),
                lon=slice(120, 280),
            )

            # Convert raw GODAS potential temperature in K to monthly anomalies.
            clim = deep_temp.groupby("time.month").mean("time")
            anom = deep_temp.groupby("time.month") - clim

            # Load into memory before the dataset is closed.
            anom = anom.load()

        return anom


    def _load_predictor_matrix(self, predictor_depth, start_date, end_date):
        """Load a predictor depth field and reshape it into a two-dimensional matrix."""
        anom = self._load_predictor_anomaly_array(predictor_depth)

        anom_sel = anom.sel(time=slice(start_date, end_date))
        arr = anom_sel.values.reshape(anom_sel.shape[0], -1)

        return arr


    def _build_nan_mask(self, predictor_depth):
        """Build a mask of grid cells that remain finite through the full anomaly record."""
        anom = self._load_predictor_anomaly_array(predictor_depth)

        arr = anom.values.reshape(anom.shape[0], -1)

        # Keep grid cells that are finite for all months.
        return np.isfinite(arr).all(axis=0)


    def data_assembly(self, date_list, predictor_depth, predictand_depth, nan_mask, lead_time):
        """Assemble full-field predictor and target arrays for a specified lead time."""
        start_date, end_date = date_list
        y_all = self._load_target_txt_series(predictand_depth)

        start_plus = pd.to_datetime(start_date) + pd.DateOffset(months=int(lead_time))
        end_plus = pd.to_datetime(end_date) + pd.DateOffset(months=int(lead_time))
        y = y_all.loc[start_plus:end_plus].copy()

        x_start = (y.index.min() - pd.DateOffset(months=int(lead_time))).strftime("%Y-%m-%d")
        x_end = (y.index.max() - pd.DateOffset(months=int(lead_time))).strftime("%Y-%m-%d")

        X = self._load_predictor_matrix(predictor_depth, x_start, x_end)[:, nan_mask]

        n = min(len(X), len(y))

        return X[:n], y.iloc[:n]

    def build_boxmean_series(self, depth, lat_range, lon_range):
        """Build a latitude-weighted box-mean anomaly time series."""
        path = os.path.join("data", f"godasClimatologyData_{int(depth)}m.nc")
        ds = xr.open_dataset(path)
        try:
            da = ds["deepTemp"].sel(
                lat=slice(lat_range[0], lat_range[1]),
                lon=slice(lon_range[0], lon_range[1]),
            )
            clim = da.groupby("time.month").mean("time")
            anom = da.groupby("time.month") - clim
            weights = xr.DataArray(
                np.cos(np.deg2rad(da["lat"].values)),
                coords={"lat": da["lat"]},
                dims=("lat",),
            )

            ts = anom.weighted(weights).mean(("lat", "lon")).to_series()
            ts.index = pd.to_datetime(ts.index)
            return ts.sort_index()
        finally:
            ds.close()

    def load_target_series(self, predictand_depth):
        """Load the target anomaly series for a selected depth."""
        return self._load_target_txt_series(predictand_depth)

    def assemble_boxmean_xy(self, predictor_series, target_series, date_list, lead_time):
        """Align a box-mean predictor with a target series at a specified lead time."""
        start_date, end_date = pd.to_datetime(date_list[0]), pd.to_datetime(date_list[1])

        y = target_series.loc[
            start_date + pd.DateOffset(months=int(lead_time)) : end_date + pd.DateOffset(months=int(lead_time))
        ].copy()
        x = predictor_series.loc[
            y.index.min() - pd.DateOffset(months=int(lead_time)) : y.index.max() - pd.DateOffset(months=int(lead_time))
        ].copy()

        n = min(len(x), len(y))
        x = x.iloc[:n]
        y = y.iloc[:n]
        return x.values.reshape(-1, 1), y

    def generate_plotly(self):
        """Generate the three-dimensional Plotly anomaly-structure figure."""
        depth_vec = [5, 35, 55, 85, 115]
        n = len(depth_vec)

        target_dates = {
            "January 1998 El Niño temperature anomaly structure": "1998-01-16",
            "January 2000 La Niña temperature anomaly structure": "2000-01-16",
        }

        lat_range = slice(-20, 20)
        lon_range = slice(120, 280)

        layer_spacing = 0.55
        z_positions = np.arange(n, dtype=float) * layer_spacing

        opacities = [1.0, 0.76, 0.72, 0.66, 0.82]

        def add_box_outline(
            traces,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            z,
            name,
            color,
            width=9,
        ):
            z_box = z - 0.08

            x = [lon_min, lon_max, lon_max, lon_min, lon_min]
            y = [lat_min, lat_min, lat_max, lat_max, lat_min]
            zz = [z_box] * 5

            traces.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=zz,
                    mode="lines",
                    line=dict(color=color, width=width),
                    name=name,
                    showlegend=False,
                    hoverinfo="name",
                )
            )

        def add_cage(
            traces,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            z_min,
            z_max,
            color="rgba(70,70,70,0.78)",
            width=4,
        ):
            corners = [
                (lon_min, lat_min, z_min),
                (lon_max, lat_min, z_min),
                (lon_max, lat_max, z_min),
                (lon_min, lat_max, z_min),
                (lon_min, lat_min, z_max),
                (lon_max, lat_min, z_max),
                (lon_max, lat_max, z_max),
                (lon_min, lat_max, z_max),
            ]

            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]

            for a, b in edges:
                traces.append(
                    go.Scatter3d(
                        x=[corners[a][0], corners[b][0]],
                        y=[corners[a][1], corners[b][1]],
                        z=[corners[a][2], corners[b][2]],
                        mode="lines",
                        line=dict(color=color, width=width),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        for title_text, date_str in target_dates.items():
            traces = []
            show_scale_for_this_fig = date_str == "2000-01-16"

            for k, depth in enumerate(depth_vec):
                path = os.path.join("data", f"godasClimatologyData_{depth}m.nc")

                if not os.path.exists(path):
                    print(f"Missing {path}")
                    continue

                ds = xr.open_dataset(path)

                try:
                    anom = to_anomaly(ds["deepTemp"])
                    temp = sel_time_nearest(anom, date_str).sel(
                        lat=lat_range,
                        lon=lon_range,
                    )

                    lats = temp["lat"].values
                    lons = temp["lon"].values
                    vals = temp.values

                    X, Y = np.meshgrid(lons, lats)
                    mask = ~np.isfinite(vals)

                    z_ocean = np.full_like(vals, z_positions[k], dtype=float)
                    z_ocean[mask] = np.nan

                    sc_ocean = np.where(mask, 0.0, vals)

                    traces.append(
                        go.Surface(
                            x=X,
                            y=Y,
                            z=z_ocean,
                            surfacecolor=sc_ocean,
                            colorscale="Jet",
                            cmin=-5,
                            cmax=5,
                            opacity=float(opacities[k]),
                            showscale=(k == 0 and show_scale_for_this_fig),
                            colorbar=dict(
                                title="°C",
                                thickness=30,
                                len=0.58,
                                y=0.50,
                                yanchor="middle",
                            ),
                            lighting=dict(
                                ambient=0.98,
                                diffuse=0.03,
                                specular=0.0,
                                roughness=1.0,
                                fresnel=0.0,
                            ),
                            lightposition=dict(x=0, y=0, z=1000),
                            showlegend=False,
                        )
                    )

                    # -----------------------------------------
                    # Grey outline around each layer
                    # -----------------------------------------

                    outline_z = z_positions[k] + 0.002

                    outline_x = [
                        lons.min(),
                        lons.max(),
                        lons.max(),
                        lons.min(),
                        lons.min(),
                    ]

                    outline_y = [
                        lats.min(),
                        lats.min(),
                        lats.max(),
                        lats.max(),
                        lats.min(),
                    ]

                    outline_zvals = [outline_z] * 5

                    traces.append(
                        go.Scatter3d(
                            x=outline_x,
                            y=outline_y,
                            z=outline_zvals,
                            mode="lines",
                            line=dict(
                                color="rgba(90,90,90,0.95)",
                                width=5,
                            ),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

                    z_land = np.full_like(vals, z_positions[k] + 1e-3, dtype=float)
                    z_land[~mask] = np.nan

                    traces.append(
                        go.Surface(
                            x=X,
                            y=Y,
                            z=z_land,
                            surfacecolor=np.zeros_like(vals),
                            colorscale=[[0, "#000000"], [1, "#000000"]],
                            cmin=0,
                            cmax=1,
                            showscale=False,
                            opacity=1.0,
                            hoverinfo="skip",
                            lighting=dict(
                                ambient=1.0,
                                diffuse=0.0,
                                specular=0.0,
                                roughness=1.0,
                                fresnel=0.0,
                            ),
                            showlegend=False,
                        )
                    )

                finally:
                    ds.close()

            z_5m = z_positions[depth_vec.index(5)]
            z_55m = z_positions[depth_vec.index(55)]


            add_box_outline(
                traces=traces,
                lat_min=-5,
                lat_max=5,
                lon_min=190,
                lon_max=240,
                z=z_5m,
                name="Niño 3.4 target box (5 m)",
                color="purple",
                width=10,
            )

            add_box_outline(
                traces=traces,
                lat_min=0,
                lat_max=10,
                lon_min=140,
                lon_max=160,
                z=z_55m,
                name="West predictor box (55 m)",
                color="white",
                width=10,
            )

            add_cage(
                traces=traces,
                lon_min=120,
                lon_max=280,
                lat_min=-20,
                lat_max=20,
                z_min=z_positions[0],
                z_max=z_positions[-1],
            )

            fig = go.Figure(traces)

            fig.update_layout(
                title=dict(
                    text=title_text,
                    font=dict(family="Arial", size=22, color="black"),
                    x=0.50,
                    y=0.93,
                    xanchor="center",
                    yanchor="top",
                ),
                template="plotly_white",
                paper_bgcolor="white",
                plot_bgcolor="white",
                margin=dict(l=10, r=10, t=40, b=10),
                font=dict(family="Arial", size=14),
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor="rgba(255,255,255,0.75)",
                    bordercolor="black",
                    borderwidth=1,
                ),
                scene_aspectmode="cube",
                scene_camera=dict(
                    up=dict(x=0, y=0, z=1),
                    eye=dict(x=1.45, y=1.25, z=0.9),
                ),
                scene=dict(
                    bgcolor="white",

                    xaxis=dict(
                        title=dict(
                            text="Longitude",
                            font=dict(size=18),
                        ),

                        range=[130, 280],
                        autorange="reversed",

                        tickvals=[130, 160, 180, 210, 240, 280],
                        ticktext=[
                            "130°E",
                            "160°E",
                            "180°",
                            "150°W",
                            "120°W",
                            "80°W",
                        ],

                        tickfont=dict(size=15),
                        ticklen=12,
                        ticks="outside",

                        backgroundcolor="rgba(225,225,225,1)",
                        gridcolor="rgba(120,120,120,0.45)",
                        zerolinecolor="rgba(80,80,80,0.7)",

                        showbackground=True,
                        showgrid=True,
                        showline=True,

                        linecolor="rgba(60,60,60,1)",
                        linewidth=3,
                    ),

                    yaxis=dict(
                        title=dict(
                            text="Latitude",
                            font=dict(size=18),
                        ),

                        tickvals=[-20, -10, 0, 10, 20],
                        ticktext=[
                            "20°S",
                            "10°S",
                            "0°",
                            "10°N",
                            "20°N",
                        ],

                        tickfont=dict(size=15),
                        ticklen=12,
                        ticks="outside",

                        autorange="reversed",

                        backgroundcolor="rgba(225,225,225,1)",
                        gridcolor="rgba(120,120,120,0.45)",
                        zerolinecolor="rgba(80,80,80,0.7)",

                        showbackground=True,
                        showgrid=True,
                        showline=True,

                        linecolor="rgba(60,60,60,1)",
                        linewidth=3,
                    ),

                    zaxis=dict(
                        title=dict(
                            text="Depth (m)",
                            font=dict(size=18),
                        ),

                        tickvals=z_positions.tolist(),
                        ticktext=[f"{d} m" for d in depth_vec],

                        tickfont=dict(size=15),
                        ticklen=12,
                        ticks="outside",

                        autorange="reversed",

                        backgroundcolor="rgba(225,225,225,1)",
                        gridcolor="rgba(120,120,120,0.55)",
                        zerolinecolor="rgba(80,80,80,0.7)",

                        showbackground=True,
                        showgrid=True,
                        showline=True,

                        linecolor="rgba(60,60,60,1)",
                        linewidth=3,
                    ),

                ),
            )

            fig.show(renderer="browser")


class EnsoLinearModel(EnsoModel):
    """Linear-regression workflow for ENSO prediction experiments and diagnostics."""
    @staticmethod
    def make_regression_features(X, feature_mode="linear"):
        """
        Create predictor features for the regression model.

        feature_mode="linear":
            X -> X

        feature_mode="quadratic":
            X -> [X, X^2]

        feature_mode="cubic":
            X -> [X, X^2, X^3]
        """
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if feature_mode == "linear":
            return X

        if feature_mode == "quadratic":
            return np.hstack([X, X**2])

        if feature_mode == "cubic":
            return np.hstack([X, X**2, X**3])

        raise ValueError(
            "feature_mode must be 'linear', 'quadratic', or 'cubic'."
        )


    @staticmethod
    def prediction_metrics(y_true, y_pred):
        """
        Compute validation metrics between truth and prediction.
        """
        y_true = pd.Series(y_true).astype(float)
        y_pred = pd.Series(y_pred, index=y_true.index).astype(float)

        valid = np.isfinite(y_true.values) & np.isfinite(y_pred.values)

        yt = y_true.values[valid]
        yp = y_pred.values[valid]

        if len(yt) < 2:
            return {
                "n": len(yt),
                "corr": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "bias": np.nan,
                "truth_std": np.nan,
                "pred_std": np.nan,
                "amp_ratio": np.nan,
            }

        corr = np.corrcoef(yt, yp)[0, 1]
        rmse = np.sqrt(np.mean((yp - yt) ** 2))
        mae = np.mean(np.abs(yp - yt))
        bias = np.mean(yp - yt)

        truth_std = np.std(yt, ddof=1)
        pred_std = np.std(yp, ddof=1)

        amp_ratio = pred_std / truth_std if truth_std != 0 else np.nan

        return {
            "n": len(yt),
            "corr": corr,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "truth_std": truth_std,
            "pred_std": pred_std,
            "amp_ratio": amp_ratio,
        }


    def run_all_scenarios(
        self,
        predictand_depth=5,
        use_deep_target=False,
        feature_mode="linear",
    ):
        """Run full-field predictor experiments against the smoothed target series."""
        month_vec = np.arange(1, 19)
        depth_vec = np.arange(5, 215, 10)
        corr_mat = np.zeros((len(month_vec), len(depth_vec)))
        rmse_mat = np.zeros((len(month_vec), len(depth_vec)))

        deep_target_series = self._load_target_txt_series(predictand_depth) if use_deep_target else None

        for mm, depth_sel in enumerate(depth_vec):
            nan_mask = self._build_nan_mask(depth_sel)

            for nn, lead in enumerate(month_vec):
                X_train, y_train = self.data_assembly(
                    self.start_stop_list_train,
                    predictor_depth=depth_sel,
                    predictand_depth=predictand_depth,
                    nan_mask=nan_mask,
                    lead_time=lead,
                )

                X_val, y_val = self.data_assembly(
                    self.start_stop_list_val,
                    predictor_depth=depth_sel,
                    predictand_depth=predictand_depth,
                    nan_mask=nan_mask,
                    lead_time=lead,
                )

                if use_deep_target:
                    start_val = pd.to_datetime(self.start_stop_list_val[0]) + pd.DateOffset(months=lead)
                    end_val = pd.to_datetime(self.start_stop_list_val[1]) + pd.DateOffset(months=lead)
                    y_val = deep_target_series.loc[start_val:end_val].iloc[: len(X_val)]

                # -----------------------------
                # linear or quadratic predictors
                # -----------------------------
                X_train_fit = self.make_regression_features(
                    X_train,
                    feature_mode=feature_mode,
                )

                X_val_fit = self.make_regression_features(
                    X_val,
                    feature_mode=feature_mode,
                )

                model = LinearRegression().fit(X_train_fit, y_train)
                y_pred = model.predict(X_val_fit)

                corr, rmse = self.corr_rmse_fcn(y_val, y_pred)
                corr_mat[nn, mm] = corr
                rmse_mat[nn, mm] = rmse

        return corr_mat, rmse_mat, month_vec, depth_vec


    def run_all_scenarios_box_target(
        self,
        target_depth=5,
        target_lat_range=(-10, 0),
        target_lon_range=(195, 225),
        feature_mode="linear",
    ):
        """
        Full-field predictor at each depth -> fixed east-box target.
        Returns corr_mat, rmse_mat, month_vec, depth_vec.
        """

        month_vec = np.arange(1, 19)
        depth_vec = np.arange(5, 215, 10)
        corr_mat = np.zeros((len(month_vec), len(depth_vec)))
        rmse_mat = np.zeros((len(month_vec), len(depth_vec)))

        target_series = self.build_boxmean_series(
            depth=target_depth,
            lat_range=target_lat_range,
            lon_range=target_lon_range,
        )

        for mm, depth_sel in enumerate(depth_vec):
            nan_mask = self._build_nan_mask(depth_sel)

            for nn, lead in enumerate(month_vec):
                X_train, _ = self.data_assembly(
                    self.start_stop_list_train,
                    predictor_depth=depth_sel,
                    predictand_depth=target_depth,
                    nan_mask=nan_mask,
                    lead_time=lead,
                )

                X_val, _ = self.data_assembly(
                    self.start_stop_list_val,
                    predictor_depth=depth_sel,
                    predictand_depth=target_depth,
                    nan_mask=nan_mask,
                    lead_time=lead,
                )

                start_train = pd.to_datetime(self.start_stop_list_train[0]) + pd.DateOffset(months=lead)
                end_train = pd.to_datetime(self.start_stop_list_train[1]) + pd.DateOffset(months=lead)
                y_train = target_series.loc[start_train:end_train].iloc[: len(X_train)]

                start_val = pd.to_datetime(self.start_stop_list_val[0]) + pd.DateOffset(months=lead)
                end_val = pd.to_datetime(self.start_stop_list_val[1]) + pd.DateOffset(months=lead)
                y_val = target_series.loc[start_val:end_val].iloc[: len(X_val)]

                ntr = min(len(X_train), len(y_train))
                X_train = X_train[:ntr]
                y_train = y_train.iloc[:ntr]

                nva = min(len(X_val), len(y_val))
                X_val = X_val[:nva]
                y_val = y_val.iloc[:nva]

                # -----------------------------
                # linear or quadratic predictors
                # -----------------------------
                X_train_fit = self.make_regression_features(
                    X_train,
                    feature_mode=feature_mode,
                )

                X_val_fit = self.make_regression_features(
                    X_val,
                    feature_mode=feature_mode,
                )

                model = LinearRegression().fit(X_train_fit, y_train)
                y_pred = model.predict(X_val_fit)

                corr, rmse = self.corr_rmse_fcn(y_val, y_pred)
                corr_mat[nn, mm] = corr
                rmse_mat[nn, mm] = rmse

        return corr_mat, rmse_mat, month_vec, depth_vec


    def compare_boxmean_polynomial_models(
        self,
        predictor_depth,
        predictand_depth,
        predictor_lat_range,
        predictor_lon_range,
        target_mode="benchmark",
        target_lat_range=None,
        target_lon_range=None,
        lead_times=(1, 3, 6, 9),
        csv_prefix=None,
    ):
        """
        Numerically compare linear, quadratic, and cubic regression models.

        No figures are produced.

        Linear:
            y = b0 + b1*x

        Quadratic:
            y = b0 + b1*x + b2*x^2

        Cubic:
            y = b0 + b1*x + b2*x^2 + b3*x^3
        """
        import os
        import numpy as np
        import pandas as pd
        import sklearn.linear_model

        feature_modes = ("linear", "quadratic", "cubic")

        # -----------------------------
        # Predictor series
        # -----------------------------
        x_series = self.build_boxmean_series(
            depth=predictor_depth,
            lat_range=predictor_lat_range,
            lon_range=predictor_lon_range,
        )

        # -----------------------------
        # Target series
        # -----------------------------
        if target_mode == "benchmark":
            y_series = self.load_target_series(predictand_depth)

        elif target_mode == "boxmean":
            if target_lat_range is None or target_lon_range is None:
                raise ValueError(
                    "For target_mode='boxmean', target_lat_range and target_lon_range must be provided."
                )

            y_series = self.build_boxmean_series(
                depth=predictand_depth,
                lat_range=target_lat_range,
                lon_range=target_lon_range,
            )

        else:
            raise ValueError("target_mode must be either 'benchmark' or 'boxmean'.")

        metric_records = []

        # =====================================================
        # Run each model for each lead time
        # =====================================================
        for lead in lead_times:
            X_tr, y_tr = self.assemble_boxmean_xy(
                x_series,
                y_series,
                self.start_stop_list_train,
                lead,
            )

            X_va, y_va = self.assemble_boxmean_xy(
                x_series,
                y_series,
                self.start_stop_list_val,
                lead,
            )

            y_va = y_va.sort_index()

            for feature_mode in feature_modes:
                X_tr_fit = self.make_regression_features(
                    X_tr,
                    feature_mode=feature_mode,
                )

                X_va_fit = self.make_regression_features(
                    X_va,
                    feature_mode=feature_mode,
                )

                model = sklearn.linear_model.LinearRegression()
                model.fit(X_tr_fit, y_tr)

                y_pred = pd.Series(
                    model.predict(X_va_fit),
                    index=y_va.index,
                ).sort_index()

                metrics = self.prediction_metrics(y_va, y_pred)

                coefs = np.asarray(model.coef_).ravel()

                metric_records.append({
                    "target_mode": target_mode,
                    "lead_months": lead,
                    "model": feature_mode,
                    "n_train": len(y_tr),
                    "n_val": metrics["n"],
                    "corr": metrics["corr"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "bias": metrics["bias"],
                    "truth_std": metrics["truth_std"],
                    "pred_std": metrics["pred_std"],
                    "amp_ratio": metrics["amp_ratio"],
                    "intercept": model.intercept_,
                    "coef_x": coefs[0] if len(coefs) > 0 else np.nan,
                    "coef_x2": coefs[1] if len(coefs) > 1 else np.nan,
                    "coef_x3": coefs[2] if len(coefs) > 2 else np.nan,
                })

        metrics_df = pd.DataFrame(metric_records)

        # =====================================================
        # Compare quadratic and cubic against linear
        # =====================================================
        comparison_records = []

        for lead in lead_times:
            lead_df = metrics_df[metrics_df["lead_months"] == lead]

            linear_row = lead_df[lead_df["model"] == "linear"].iloc[0]

            for model_name in ("quadratic", "cubic"):
                model_row = lead_df[lead_df["model"] == model_name].iloc[0]

                comparison_records.append({
                    "target_mode": target_mode,
                    "lead_months": lead,
                    "model_compared_to_linear": model_name,

                    "linear_corr": linear_row["corr"],
                    f"{model_name}_corr": model_row["corr"],
                    "delta_corr": model_row["corr"] - linear_row["corr"],

                    "linear_rmse": linear_row["rmse"],
                    f"{model_name}_rmse": model_row["rmse"],
                    "delta_rmse": model_row["rmse"] - linear_row["rmse"],

                    "linear_mae": linear_row["mae"],
                    f"{model_name}_mae": model_row["mae"],
                    "delta_mae": model_row["mae"] - linear_row["mae"],

                    "linear_bias": linear_row["bias"],
                    f"{model_name}_bias": model_row["bias"],
                    "delta_bias": model_row["bias"] - linear_row["bias"],

                    "linear_amp_ratio": linear_row["amp_ratio"],
                    f"{model_name}_amp_ratio": model_row["amp_ratio"],
                    "delta_amp_ratio": model_row["amp_ratio"] - linear_row["amp_ratio"],
                })

        comparison_df = pd.DataFrame(comparison_records)

        print("\n=====================================================")
        print(f"Polynomial model metrics: {target_mode}")
        print("=====================================================")
        print(metrics_df.round(4).to_string(index=False))

        print("\n=====================================================")
        print(f"Quadratic/cubic minus linear comparison: {target_mode}")
        print("Positive delta_corr is good. Negative delta_rmse is good.")
        print("=====================================================")
        print(comparison_df.round(4).to_string(index=False))

        if csv_prefix is not None:
            os.makedirs("stats", exist_ok=True)

            metrics_path = os.path.join("stats", f"{csv_prefix}_metrics.csv")
            comparison_path = os.path.join("stats", f"{csv_prefix}_comparison.csv")

            metrics_df.to_csv(metrics_path, index=False)
            comparison_df.to_csv(comparison_path, index=False)

            print(f"\nSaved metrics to: {metrics_path}")
            print(f"Saved comparison to: {comparison_path}")

        return metrics_df, comparison_df


    def generate_boxmean_prediction_figure(
        self,
        predictor_depth,
        predictand_depth,
        predictor_lat_range,
        predictor_lon_range,
        figure_title,
        panel_title_prefix,
        ylabel,
        out_name,
        target_mode="benchmark",      # "benchmark" or "boxmean"
        target_lat_range=None,
        target_lon_range=None,
        lead_times=(1, 3, 6, 9),
        y_limits=None,
        feature_mode="quadratic",        # "linear" or "quadratic"
    ):
        """Plot observed and regression-predicted box-mean ENSO time series."""
        import matplotlib.ticker as mticker
        import matplotlib.dates as mdates
        import sklearn.linear_model

        if feature_mode not in ("linear", "quadratic", "cubic"):
            raise ValueError("feature_mode must be either 'linear' or 'quadratic' or 'cubic'.")

        # -----------------------------
        # predictor series: always box mean
        # -----------------------------
        x_series = self.build_boxmean_series(
            depth=predictor_depth,
            lat_range=predictor_lat_range,
            lon_range=predictor_lon_range,
        )

        # -----------------------------
        # target series: either benchmark txt or box mean
        # -----------------------------
        if target_mode == "benchmark":
            y_series = self.load_target_series(predictand_depth)

        elif target_mode == "boxmean":
            if target_lat_range is None or target_lon_range is None:
                raise ValueError(
                    "For target_mode='boxmean', target_lat_range and target_lon_range must be provided."
                )

            y_series = self.build_boxmean_series(
                depth=predictand_depth,
                lat_range=target_lat_range,
                lon_range=target_lon_range,
            )

        else:
            raise ValueError("target_mode must be either 'benchmark' or 'boxmean'.")

        fig, axs = plt.subplots(
            len(lead_times), 1,
            figsize=(12, 10),
            sharex=False,
            constrained_layout=False,
        )

        main_col = axs[0].get_position(fig)
        x_center = (main_col.x0 + main_col.x1) / 2

        fig.suptitle(
            figure_title,
            fontsize=18,
            fontweight="bold",
            x=x_center,
            ha="center",
            y=0.99,
        )

        fig.subplots_adjust(
            top=0.92,
            bottom=0.07,
            left=0.08,
            right=0.98,
            hspace=0.4,
        )

        labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

        if y_limits is None:
            y_min, y_max = np.inf, -np.inf
        else:
            y_min, y_max = y_limits

        cached = []

        for lead in lead_times:
            X_tr, y_tr = self.assemble_boxmean_xy(
                x_series,
                y_series,
                self.start_stop_list_train,
                lead,
            )

            X_va, y_va = self.assemble_boxmean_xy(
                x_series,
                y_series,
                self.start_stop_list_val,
                lead,
            )

            # -----------------------------
            # linear or quadratic features
            # -----------------------------
            X_tr_fit = self.make_regression_features(
                X_tr,
                feature_mode=feature_mode,
            )

            X_va_fit = self.make_regression_features(
                X_va,
                feature_mode=feature_mode,
            )

            # -----------------------------
            # regression
            # -----------------------------
            regr = sklearn.linear_model.LinearRegression()
            regr.fit(X_tr_fit, y_tr)

            y_hat = pd.Series(
                regr.predict(X_va_fit),
                index=y_va.index,
            ).sort_index()

            y_va = y_va.sort_index()

            cached.append((lead, y_va, y_hat))

            if y_limits is None:
                y_min = min(
                    y_min,
                    float(np.nanmin([y_va.min(), y_hat.min()])),
                )
                y_max = max(
                    y_max,
                    float(np.nanmax([y_va.max(), y_hat.max()])),
                )

        if y_limits is None:
            pad = 0.15 * (y_max - y_min + 1e-9)
            y_min -= pad
            y_max += pad

        for i, (lead, y_va, y_hat) in enumerate(cached):
            ax = axs[i]

            ax.plot(y_va, label="Observed", linewidth=2.2, color="blue")
            ax.plot(y_hat, label="Regression prediction", linewidth=2.2, color="red")

            ax.set_title(
                f"{panel_title_prefix}: {lead}-month lead",
                fontsize=14,
            )

            ax.set_ylabel(ylabel, fontsize=12)

            ax.set_xlim(y_va.index[0], y_va.index[-1])
            ax.margins(x=0)

            ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

            ax.set_ylim(y_min, y_max)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(1))

            ax.grid(True, linestyle=" ", linewidth=0.7, alpha=0.6)
            ax.tick_params(labelsize=11)

            ax.text(
                -0.06,
                1.02,
                labels[i],
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=15,
                fontweight="bold",
            )

            ax.legend(
                loc="upper right",
                bbox_to_anchor=(0.95, 1.0),
                frameon=False,
                fontsize=12.4,
                borderaxespad=0,
            )

        axs[-1].set_xlabel("Year", fontsize=14)

        plt.savefig(
            os.path.join("figures", out_name),
            dpi=400,
            bbox_inches="tight",
        )

        plt.show()

    def run_boxmean_heatmap_scenarios(
        self,
        predictor_lat_range,
        predictor_lon_range,
        target_lat_range,
        target_lon_range,
        target_depth=5,
        lead_times=np.arange(1, 19),
        depth_vec=np.arange(5, 215, 10),
    ):
        """
        Build correlation and RMSE heatmaps for:
            west box at varying predictor depths -> fixed east box target at target_depth
        """

        month_vec = np.array(lead_times)
        depth_vec = np.array(depth_vec)

        corr_mat = np.zeros((len(month_vec), len(depth_vec)))
        rmse_mat = np.zeros((len(month_vec), len(depth_vec)))

        # fixed east target series
        y_series = self.build_boxmean_series(
            depth=target_depth,
            lat_range=target_lat_range,
            lon_range=target_lon_range,
        )

        for mm, depth_sel in enumerate(depth_vec):
            # west predictor series at this depth
            x_series = self.build_boxmean_series(
                depth=depth_sel,
                lat_range=predictor_lat_range,
                lon_range=predictor_lon_range,
            )

            for nn, lead in enumerate(month_vec):
                X_train, y_train = self.assemble_boxmean_xy(
                    x_series, y_series, self.start_stop_list_train, lead
                )
                X_val, y_val = self.assemble_boxmean_xy(
                    x_series, y_series, self.start_stop_list_val, lead
                )

                model = LinearRegression().fit(X_train, y_train)
                y_pred = model.predict(X_val)

                corr, rmse = self.corr_rmse_fcn(y_val, y_pred)
                corr_mat[nn, mm] = corr
                rmse_mat[nn, mm] = rmse

        return corr_mat, rmse_mat, month_vec, depth_vec

    def run_fixed_predictor_box_to_target_depth_heatmap(
        self,
        predictor_depth,
        predictor_lat_range,
        predictor_lon_range,
        target_lat_range,
        target_lon_range,
        lead_times=np.arange(1, 19),
        target_depth_vec=np.arange(5, 215,10),
    ):
        """
        Fix predictor = one west box at one depth (e.g. 135 m),
        and sweep target depth + lead time.

        Returns corr_mat, rmse_mat, month_vec, target_depth_vec.
        """

        month_vec = np.array(lead_times)
        target_depth_vec = np.array(target_depth_vec)

        corr_mat = np.zeros((len(month_vec), len(target_depth_vec)))
        rmse_mat = np.zeros((len(month_vec), len(target_depth_vec)))

        # fixed predictor series
        x_series = self.build_boxmean_series(
            depth=predictor_depth,
            lat_range=predictor_lat_range,
            lon_range=predictor_lon_range,
        )

        for mm, target_depth in enumerate(target_depth_vec):
            y_series = self.build_boxmean_series(
                depth=target_depth,
                lat_range=target_lat_range,
                lon_range=target_lon_range,
            )

            for nn, lead in enumerate(month_vec):
                X_train, y_train = self.assemble_boxmean_xy(
                    x_series, y_series, self.start_stop_list_train, lead
                )
                X_val, y_val = self.assemble_boxmean_xy(
                    x_series, y_series, self.start_stop_list_val, lead
                )

                model = LinearRegression().fit(X_train, y_train)
                y_pred = model.predict(X_val)

                corr, rmse = self.corr_rmse_fcn(y_val, y_pred)
                corr_mat[nn, mm] = corr
                rmse_mat[nn, mm] = rmse

        return corr_mat, rmse_mat, month_vec, target_depth_vec

# Heatmaps for Figures 7 & 8

def corr_paper_cmap():
    """Return the custom correlation colormap used in manuscript heatmaps."""
    return LinearSegmentedColormap.from_list(
        "corr_paper",
        [
            (0.0, "#2c00ff"),
            (0.25, "#00f6ff"),
            (0.5, "#ffffff"),
            (0.75, "#fff800"),
            (1.0, "#ff0000"),
        ],
    )


def plot_correlation_heatmap(
    corr_mat,
    month_vec,
    depth_vec,
    fig_path,
    title="Full-field prediction skill: correlation",
):
    """Plot a depth-by-lead correlation heatmap."""
    fig, ax = plt.subplots(figsize=(8.5, 7))

    # corr_mat is shape: (lead, depth)
    # transpose gives: (depth, lead)
    # origin="upper" puts 5 m at the top
    data = corr_mat.T

    corr_step = 0.05   # smaller value = more bins

    corr_bounds = np.round(
        np.arange(0.0, 1.0 + corr_step, corr_step),
        2
    )

    corr_cmap = corr_paper_cmap().resampled(len(corr_bounds) - 1)
    corr_norm = BoundaryNorm(corr_bounds, ncolors=corr_cmap.N, clip=True)

    im = ax.imshow(
        data,
        aspect="auto",
        origin="upper",
        cmap=corr_cmap,
        norm=corr_norm,
    )

    ax.set_xticks(np.arange(len(month_vec)))
    ax.set_xticklabels(month_vec, fontsize=15)

    ax.set_yticks(np.arange(len(depth_vec)))
    ax.set_yticklabels(depth_vec, fontsize=15)

    ax.set_xlabel("Forecast Lead Time (months)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title, fontsize=16, pad=10, fontweight="bold")

    cbar = plt.colorbar(
        im,
        ax=ax,
        shrink=0.95,
        pad=0.015,
        boundaries=corr_bounds,
        ticks=corr_bounds,
        spacing="uniform",
        drawedges=True,
    )
    cbar.set_label("Correlation", fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=16, width=1.2, length=6)

    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.grid(False)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=400, bbox_inches="tight")
    plt.show()


def plot_rmse_heatmap(
    rmse_mat,
    month_vec,
    depth_vec,
    fig_path,
    title="Full-field prediction skill: RMSE",
    title_size=16,
):
    """Plot a depth-by-lead RMSE heatmap."""
    fig, ax = plt.subplots(figsize=(8.5, 7))

    # rmse_mat is shape: (lead, depth)
    data = rmse_mat.T

    rmse_step = 0.10
    rmse_max = float(np.nanmax(data))
    rmse_upper = rmse_step * np.ceil(rmse_max / rmse_step)
    rmse_upper = max(rmse_upper, rmse_step)

    rmse_bounds = np.arange(0.0, rmse_upper + rmse_step * 0.5, rmse_step)
    rmse_cmap = plt.get_cmap("Reds", len(rmse_bounds) - 1)
    rmse_norm = BoundaryNorm(rmse_bounds, ncolors=rmse_cmap.N, clip=True)

    im = ax.imshow(
        data,
        aspect="auto",
        origin="upper",
        cmap=rmse_cmap,
        norm=rmse_norm,
    )

    ax.set_xticks(np.arange(len(month_vec)))
    ax.set_xticklabels(month_vec, fontsize=15, fontweight="bold")

    ax.set_yticks(np.arange(len(depth_vec)))
    ax.set_yticklabels(depth_vec, fontsize=15, fontweight="bold")

    ax.set_xlabel("Forecast Lead Time (months)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Depth (m)", fontsize=18, fontweight="bold")
    ax.set_title(title, fontsize=16, pad=10, fontweight="bold")

    cbar = plt.colorbar(
        im,
        ax=ax,
        shrink=0.95,
        pad=0.015,
        boundaries=rmse_bounds,
        ticks=rmse_bounds,
        spacing="uniform",
        drawedges=True,
    )
    cbar.set_label("RMSE", fontsize=16, fontweight="bold")
    cbar.ax.set_title("°C", fontsize=16, fontweight="bold", pad=6)
    cbar.ax.tick_params(labelsize=16, width=1.2, length=6)

    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.grid(False)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=400, bbox_inches="tight")
    plt.show()


def save_polynomial_comparison_csv(metrics_df, out_path):
    """
    Save a clean side-by-side CSV comparing linear, quadratic, and cubic models.
    Each row is one lead time.
    """
    import os
    import pandas as pd

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    metrics_to_compare = [
        "corr",
        "rmse",
        "mae",
        "bias",
        "amp_ratio",
    ]

    keep_cols = ["lead_months", "model"] + metrics_to_compare
    df = metrics_df[keep_cols].copy()

    wide_df = df.pivot(
        index="lead_months",
        columns="model",
        values=metrics_to_compare,
    )

    wide_df.columns = [
        f"{model}_{metric}"
        for metric, model in wide_df.columns
    ]

    wide_df = wide_df.reset_index()

    for metric in metrics_to_compare:
        wide_df[f"quadratic_minus_linear_{metric}"] = (
            wide_df[f"quadratic_{metric}"] - wide_df[f"linear_{metric}"]
        )

        wide_df[f"cubic_minus_linear_{metric}"] = (
            wide_df[f"cubic_{metric}"] - wide_df[f"linear_{metric}"]
        )

    best_models = []

    for _, row in wide_df.iterrows():
        rmse_values = {
            "linear": row["linear_rmse"],
            "quadratic": row["quadratic_rmse"],
            "cubic": row["cubic_rmse"],
        }

        best_model = min(rmse_values, key=rmse_values.get)
        best_models.append(best_model)

    wide_df["best_model_by_rmse"] = best_models

    wide_df = wide_df.round(4)

    wide_df.to_csv(out_path, index=False)

    print("\nSaved side-by-side polynomial comparison CSV to:")
    print(out_path)

    print("\nSide-by-side polynomial comparison:")
    print(wide_df.to_string(index=False))

    return wide_df


if __name__ == "__main__":
    start = time.time()
    _ensure_output_dirs()

    # Standalone retained figures
    generate_deep_to_surface_event_maps()

    obj = EnsoLinearModel()

    # metrics_smoothed_poly, comparison_smoothed_poly = obj.compare_boxmean_polynomial_models(
    #     predictor_depth=55,
    #     predictand_depth=5,
    #     predictor_lat_range=(0, 10),
    #     predictor_lon_range=(140, 160),
    #     target_mode="benchmark",
    #     lead_times=(1, 3, 6, 9),
    #     csv_prefix="smoothed_linear_quadratic_cubic",
    # )

    # comparison_table_4a = save_polynomial_comparison_csv(
    #     metrics_df=metrics_smoothed_poly,
    #     out_path=os.path.join("stats", "smoothed_polynomial_side_by_side_comparison.csv"),
    # )

    # metrics_raw_poly, comparison_raw_poly = obj.compare_boxmean_polynomial_models(
    #     predictor_depth=55,
    #     predictand_depth=5,
    #     predictor_lat_range=(0, 10),
    #     predictor_lon_range=(140, 160),
    #     target_mode="boxmean",
    #     target_lat_range=(-5, 5),
    #     target_lon_range=(190, 240),
    #     lead_times=(1, 3, 6, 9),
    #     csv_prefix="raw_linear_quadratic_cubic",
    # )

    # comparison_table_4b = save_polynomial_comparison_csv(
    #     metrics_df=metrics_raw_poly,
    #     out_path=os.path.join("stats", "raw_polynomial_side_by_side_comparison.csv"),
    # )

    # Figure 1
    obj.generate_plotly()

    # Boxmean figures
    # Nino 3.4 region 5N-5S, 170W-120W

    obj.generate_boxmean_prediction_figure(
        predictor_depth=55,
        predictand_depth=5,
        predictor_lat_range=(0, 10),
        predictor_lon_range=(140, 160),
        target_mode="benchmark",
        figure_title="Observed and Regression-Predicted Smoothed Niño 3.4 Anomalies",
        panel_title_prefix="Smoothed Niño 3.4 anomaly",
        ylabel="Smoothed Niño 3.4 anomaly",
        out_name="fig_6.png",
        y_limits=(-3, 3.5),
        feature_mode="linear",
    )

    obj.generate_boxmean_prediction_figure(
        predictor_depth=55,
        predictand_depth=5,
        predictor_lat_range=(0, 10),
        predictor_lon_range=(140, 160),
        target_mode="boxmean",
        target_lat_range=(-5, 5),
        target_lon_range=(190, 240),
        figure_title="Observed and Regression-Predicted GODAS Niño 3.4 Anomalies",
        panel_title_prefix="GODAS Niño 3.4 anomaly",
        ylabel="GODAS Niño 3.4 anomaly",
        out_name="fig_7.png",
        y_limits=(-3, 3.5),
        feature_mode="linear",
    )


    # Figure 4
    try:
        corr4 = np.load(os.path.join("stats", "corr_fig4.npy"))
        rmse4 = np.load(os.path.join("stats", "rmse_fig4.npy"))
        months4 = np.load(os.path.join("stats", "months_fig4.npy"))
        depths4 = np.load(os.path.join("stats", "depths_fig4.npy"))
    except Exception:
        corr4, rmse4, months4, depths4 = obj.run_all_scenarios(
            predictand_depth=5,
            use_deep_target=False
        )
        np.save(os.path.join("stats", "corr_fig4.npy"), corr4)
        np.save(os.path.join("stats", "rmse_fig4.npy"), rmse4)
        np.save(os.path.join("stats", "months_fig4.npy"), months4)
        np.save(os.path.join("stats", "depths_fig4.npy"), depths4)
        print("Computed and saved data for Figure 4.")

    plot_correlation_heatmap(
        corr4,
        months4,
        depths4,
        "figures/fig_4a.png",
        title="Correlation: smoothed Niño 3.4 target",
    )

    plot_rmse_heatmap(
        rmse4,
        months4,
        depths4,
        "figures/fig_4b.png",
        title="RMSE: smoothed Niño 3.4 target",
    )

    # Figure 5: full-field predictor depths -> Niño 3.4 surface-box target
    required_depths = np.arange(5, 215, 10)

    try:
        corr5 = np.load(os.path.join("stats", "corr_fig5.npy"))
        rmse5 = np.load(os.path.join("stats", "rmse_fig5.npy"))
        months5 = np.load(os.path.join("stats", "months_fig5.npy"))
        depths5 = np.load(os.path.join("stats", "depths_fig5.npy"))

        if not np.array_equal(depths5, required_depths):
            raise ValueError("Cached Figure 5 depths are stale.")

        print("Loaded cached data for Figure 5.")

    except Exception:
        corr5, rmse5, months5, depths5 = obj.run_all_scenarios_box_target(
            target_depth=5,
            target_lat_range=(-5, 5),
            target_lon_range=(190, 240),
        )

        np.save(os.path.join("stats", "corr_fig5.npy"), corr5)
        np.save(os.path.join("stats", "rmse_fig5.npy"), rmse5)
        np.save(os.path.join("stats", "months_fig5.npy"), months5)
        np.save(os.path.join("stats", "depths_fig5.npy"), depths5)

        print("Computed and saved data for Figure 5.")

    plot_correlation_heatmap(
        corr5,
        months5,
        depths5,
        os.path.join("figures", "fig_5a.png"),
        title="Correlation: GODAS Niño 3.4 target",
    )

    plot_rmse_heatmap(
        rmse5,
        months5,
        depths5,
        os.path.join("figures", "fig_5b.png"),
        title="RMSE: GODAS Niño 3.4 target",
    )


    print(f"Finished in: {round(time.time() - start)} seconds")
