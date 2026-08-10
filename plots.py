"""
Plotting and display helpers for the climate dashboard.

Fix for Figures 2-5:
- remove old dotted vertical forecast-start markers
- clip dashed ECMWF forecast median lines so they only appear over real forecast months
- break dashed lines at month gaps, so no dashed line crosses non-forecast months
- deduplicate legends before Streamlit renders captured figures

Save this file as plots.py in the same folder as app.py.
"""
from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.collections import PolyCollection


def show_dataframe(obj: Any) -> None:
    """Show pandas DataFrames and Styler objects cleanly in Streamlit."""
    if isinstance(obj, pd.io.formats.style.Styler):
        st.dataframe(obj.data, use_container_width=True)
    elif isinstance(obj, pd.DataFrame):
        st.dataframe(obj, use_container_width=True)
    else:
        st.write(obj)


def metric_card(label: str, value: Any, help_text: Optional[str] = None) -> None:
    """Small wrapper around st.metric with NA fallback."""
    st.metric(label, "NA" if value is None else value, help=help_text)


def _as_float_array(values: Any) -> np.ndarray:
    try:
        return np.asarray(values, dtype=float)
    except Exception:
        return np.array([], dtype=float)


def _dedupe_axis_legend(ax: Any) -> None:
    """Remove duplicate legend labels while keeping the first occurrence."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles or not labels:
        return

    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and not str(label).startswith("_") and label not in unique:
            unique[label] = handle

    if unique:
        ax.legend(unique.values(), unique.keys(), fontsize=8, loc="best")


def _remove_existing_vertical_markers(ax: Any) -> None:
    """Remove constant-x dotted/dashed vertical forecast markers from an axis."""
    for line in list(ax.lines):
        x = _as_float_array(line.get_xdata())
        y = _as_float_array(line.get_ydata())

        if x.size < 2 or y.size < 2 or not np.isfinite(x).any():
            continue

        finite_x = x[np.isfinite(x)]
        if finite_x.size < 2:
            continue

        is_constant_x = np.nanmax(finite_x) - np.nanmin(finite_x) < 1e-9
        is_marker_style = line.get_linestyle() in {":", "--", "-."}
        is_blackish = (
            line.get_color() in {"black", "k"}
            or str(line.get_color()).lower() in {"#000000", "0"}
        )

        if is_constant_x and is_marker_style and is_blackish:
            try:
                line.remove()
            except Exception:
                pass


def _forecast_x_range_from_envelopes(ax: Any) -> tuple[float, float] | None:
    """
    Infer the real forecast x-range from the black/grey uncertainty envelopes.

    In the pipeline, ECMWF IQR and 5-95% bands are black/grey fill_between
    PolyCollections. Their x-coordinates are the safest indicator of the actual
    forecast months present on the plot.
    """
    xs: list[np.ndarray] = []

    for collection in ax.collections:
        if not isinstance(collection, PolyCollection):
            continue

        facecolors = collection.get_facecolors()
        if facecolors is None or len(facecolors) == 0:
            continue

        r, g, b, _a = facecolors[0][:4]
        is_grey_or_black = max(r, g, b) - min(r, g, b) < 0.08 and max(r, g, b) < 0.30
        if not is_grey_or_black:
            continue

        for path in collection.get_paths():
            vertices = path.vertices
            if vertices is None or len(vertices) == 0:
                continue
            x = vertices[:, 0]
            x = x[np.isfinite(x)]
            if x.size:
                xs.append(x)

    if not xs:
        return None

    all_x = np.concatenate(xs)
    if all_x.size == 0:
        return None

    return float(np.nanmin(all_x)), float(np.nanmax(all_x))


def _break_line_at_month_gaps(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert NaNs where a calendar-month line jumps over a gap.

    This prevents matplotlib from drawing a dashed forecast line across months
    where there is no forecast, including the Dec-Jan wrap.
    """
    if x.size <= 1:
        return x, y

    new_x: list[float] = [float(x[0])]
    new_y: list[float] = [float(y[0])]

    for i in range(1, x.size):
        previous_x = x[i - 1]
        current_x = x[i]

        if np.isfinite(previous_x) and np.isfinite(current_x):
            # Consecutive calendar months should differ by roughly one x step.
            if abs(current_x - previous_x) > 1.01:
                new_x.append(np.nan)
                new_y.append(np.nan)

        new_x.append(float(current_x))
        new_y.append(float(y[i]))

    return np.asarray(new_x, dtype=float), np.asarray(new_y, dtype=float)


def _clip_forecast_median_lines_to_forecast_months(ax: Any) -> None:
    """Clip dashed ECMWF median lines to the actual forecast months only."""
    forecast_range = _forecast_x_range_from_envelopes(ax)
    if forecast_range is None:
        return

    xmin, xmax = forecast_range

    for line in list(ax.lines):
        x = _as_float_array(line.get_xdata())
        y = _as_float_array(line.get_ydata())

        if x.size < 2 or y.size < 2 or x.size != y.size:
            continue

        linestyle = line.get_linestyle()
        label = str(line.get_label()).lower()
        color = str(line.get_color()).lower()

        is_labelled_forecast_median = (
            linestyle in {"--", ":", "-."}
            and "forecast" in label
            and "median" in label
        )

        # Fallback if the captured figure lost the label.
        is_black_dashed_multi_point = (
            linestyle in {"--", ":", "-."}
            and color in {"black", "k", "#000000"}
            and np.unique(x[np.isfinite(x)]).size > 1
        )

        if not (is_labelled_forecast_median or is_black_dashed_multi_point):
            continue

        keep = np.isfinite(x) & np.isfinite(y) & (x >= xmin - 1e-6) & (x <= xmax + 1e-6)
        if keep.sum() == 0:
            try:
                line.remove()
            except Exception:
                pass
            continue

        x_new = x[keep]
        y_new = y[keep]
        x_new, y_new = _break_line_at_month_gaps(x_new, y_new)

        line.set_xdata(x_new)
        line.set_ydata(y_new)


def fix_forecast_marker_and_legend(fig: Any, figure_number: int | None = None) -> Any:
    """
    Clean captured matplotlib figures before Streamlit renders them.

    Applies mainly to Figures 2-5, but it is safe to run on all figures.
    """
    if fig is None or not hasattr(fig, "axes"):
        return fig

    for ax in fig.axes:
        _remove_existing_vertical_markers(ax)
        _clip_forecast_median_lines_to_forecast_months(ax)
        _dedupe_axis_legend(ax)

    try:
        fig.canvas.draw_idle()
    except Exception:
        pass

    return fig


def render_captured_figure(figures: list[Any]) -> None:
    """Render one captured matplotlib figure selected with a slider."""
    if not figures:
        st.info("No matplotlib figures were captured. Check whether the source workflow calls plt.show().")
        return

    idx = st.slider("Captured figure", 1, len(figures), 1) - 1
    st.pyplot(fix_forecast_marker_and_legend(figures[idx], idx + 1), clear_figure=False)


def _first_matching_column(df: pd.DataFrame, required_terms: list[str]) -> str | None:
    """Find the first column whose name contains all required terms."""
    for col in df.columns:
        lower = str(col).lower()
        if all(term in lower for term in required_terms):
            return col
    return None


def render_monthly_forecast_chart(df: pd.DataFrame, region_name: str) -> None:
    """Draw a simple two-panel monthly forecast chart from a regional forecast table."""
    if df is None or df.empty:
        st.info("No data available for this region.")
        return

    temp_col = _first_matching_column(df, ["temperature", "anomaly"])
    precip_col = _first_matching_column(df, ["precipitation", "anomaly"])
    month_col = "Month" if "Month" in df.columns else df.columns[0]

    if temp_col is None or precip_col is None:
        show_dataframe(df)
        return

    plot_df = df.copy()
    x = plot_df[month_col].astype(str)

    fig, axes = plt.subplots(ncols=2, figsize=(11, 3.6))
    axes[0].bar(x, plot_df[temp_col].astype(float), color="firebrick", alpha=0.75)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title(f"{region_name} temperature")
    axes[0].set_ylabel("Deg C anomaly")

    axes[1].bar(x, plot_df[precip_col].astype(float), color="royalblue", alpha=0.75)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(f"{region_name} precipitation")
    axes[1].set_ylabel("mm/month anomaly")

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_provider_bar_chart(provider_summary: pd.DataFrame) -> None:
    """Render a compact provider-comparison chart if the expected columns exist."""
    if provider_summary is None or provider_summary.empty:
        st.info("No provider-comparison data available.")
        return

    show_dataframe(provider_summary)


def render_download_button_for_dataframe(df: pd.DataFrame, filename: str, label: str) -> None:
    """Add a CSV download button for a DataFrame."""
    if df is None or df.empty:
        return

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
    )
