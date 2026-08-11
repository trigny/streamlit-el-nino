"""
Streamlit app for the El Nino climate dashboard.

Tabs:
- Overview
- Peru
- Madagascar
- Thailand
- Kenya
- Switzerland

Run with:
python -m streamlit run app.py
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from climate_pipeline import (
    DEFAULT_DATA_DIR_NAME,
    WORKFLOW_HASH,
    format_month,
    get_dataframe,
    run_pipeline,
)
from plots import (
    render_download_button_for_dataframe,
    show_dataframe,
)

try:
    from plots import fix_forecast_marker_and_legend
except Exception:
    def fix_forecast_marker_and_legend(fig: Any, figure_number: int | None = None) -> Any:
        return fig


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / DEFAULT_DATA_DIR_NAME

MAX_FIGURE_WIDTH_PX = 1100
FIGURE_RENDER_DPI = 145


# =====================================================================
# General rendering helpers
# =====================================================================

def render_limited_width_matplotlib_figure(
    fig: Any,
    max_width_px: int = MAX_FIGURE_WIDTH_PX,
) -> None:
    """
    Render a matplotlib figure with a maximum display width.

    Streamlit's normal st.pyplot output expands with the page width, which can
    make figures too large on full-screen or large external monitors. Rendering
    the figure as an inline PNG inside a max-width HTML container keeps the
    figure readable and centered.
    """
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=FIGURE_RENDER_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")

    html = f"""
    <div style="width:100%; display:flex; justify-content:center; margin-top:0.5rem; margin-bottom:0.5rem;">
        <img src="data:image/png;base64,{encoded}"
             style="max-width:{max_width_px}px; width:100%; height:auto; display:block;" />
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_figure_title(fig: Any, fallback: str = "Workflow figure") -> str:
    """Return a readable title from a matplotlib figure."""
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        try:
            text = str(suptitle.get_text()).strip()
            if text:
                return text
        except Exception:
            pass

    for ax in getattr(fig, "axes", []):
        try:
            text = str(ax.get_title()).strip()
            if text:
                return text
        except Exception:
            pass

    return fallback


def compact_metric(label: str, value: Any) -> None:
    value_text = "NA" if value is None else str(value)
    st.markdown(
        f"""
        <div style="padding:0.7rem 0.8rem; border:1px solid #e6e8eb; border-radius:0.5rem; background:#ffffff; min-height:5.1rem;">
            <div style="font-size:0.78rem; color:#6b7280; margin-bottom:0.35rem;">{label}</div>
            <div style="font-size:1.0rem; font-weight:650; color:#111827; line-height:1.25;">{value_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_figure(
    figures: list[Any],
    figure_number: int,
    title: str | None = None,
    missing_message: str | None = None,
) -> None:
    """
    Render one captured workflow figure without exposing legacy figure numbers
    in the UI.
    """
    index = figure_number - 1

    if not (0 <= index < len(figures)):
        if missing_message:
            st.info(missing_message)
        return

    fig = fix_forecast_marker_and_legend(figures[index], figure_number)
    heading = title or get_figure_title(fig)
    st.markdown(f"### {heading}")
    render_limited_width_matplotlib_figure(fig)
    st.caption("Captured workflow output.")


# =====================================================================
# Pipeline metadata helpers
# =====================================================================

def get_forecast_label(namespace: dict[str, Any]) -> str:
    if namespace.get("FORECAST_LABEL"):
        return str(namespace["FORECAST_LABEL"])

    configs = namespace.get("FORECAST_CONFIGS", {})
    primary = namespace.get("PRIMARY_FORECAST_SOURCE")

    if isinstance(configs, dict) and primary in configs:
        return str(configs[primary].get("label", primary))

    return "NA"


def infer_historical_start(namespace: dict[str, Any]) -> str:
    start_year = namespace.get("ERA5_START_YEAR")
    if start_year is not None:
        try:
            return f"{int(start_year):04d}-01"
        except Exception:
            pass

    for name in ["nino34_index", "nino34_running", "oni_series"]:
        obj = namespace.get(name)
        try:
            if hasattr(obj, "time"):
                return pd.Timestamp(obj.time.min().values).strftime("%Y-%m")
            if hasattr(obj, "index") and len(obj.index) > 0:
                return pd.Timestamp(obj.index.min()).strftime("%Y-%m")
        except Exception:
            continue

    return "NA"


def format_range(start_value: Any, end_value: Any) -> str:
    start = format_month(start_value)
    end = format_month(end_value)

    if start == "NA" and end == "NA":
        return "NA"
    if start == "NA":
        return f"to {end}"
    if end == "NA":
        return f"from {start}"
    return f"{start} to {end}"


# =====================================================================
# Figure matching helpers
# =====================================================================

def figure_text_contains(fig: Any, required_terms: list[str]) -> bool:
    """Return True if all required terms are present in figure text."""
    texts: list[str] = []

    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        try:
            texts.append(str(suptitle.get_text()))
        except Exception:
            pass

    for ax in getattr(fig, "axes", []):
        for getter in [ax.get_title, ax.get_xlabel, ax.get_ylabel]:
            try:
                texts.append(str(getter()))
            except Exception:
                pass

        try:
            _, labels = ax.get_legend_handles_labels()
            texts.extend([str(label) for label in labels])
        except Exception:
            pass

    joined = " ".join(texts).lower()
    return all(term.lower() in joined for term in required_terms)


def find_figures_by_terms(figures: list[Any], required_terms: list[str]) -> list[int]:
    matches: list[int] = []

    for i, fig in enumerate(figures, start=1):
        try:
            if figure_text_contains(fig, required_terms):
                matches.append(i)
        except Exception:
            pass

    return matches


def figure_matches_region(fig: Any, region_name: str) -> bool:
    return figure_text_contains(fig, [region_name])


def find_region_figures(figures: list[Any], region_name: str) -> list[int]:
    matches: list[int] = []

    for i, fig in enumerate(figures, start=1):
        try:
            if figure_matches_region(fig, region_name):
                matches.append(i)
        except Exception:
            pass

    return matches


def unique_existing_figure_numbers(numbers: list[int], figures: list[Any]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()

    for n in numbers:
        if 1 <= n <= len(figures) and n not in seen:
            out.append(n)
            seen.add(n)

    return out


# =====================================================================
# Dataframe helpers
# =====================================================================

def find_region_table(namespace: dict[str, Any], region_name: str) -> pd.DataFrame | None:
    forecast_tables = namespace.get("forecast_tables")

    if isinstance(forecast_tables, dict):
        if region_name in forecast_tables and isinstance(forecast_tables[region_name], pd.DataFrame):
            return forecast_tables[region_name].copy()

        for key, value in forecast_tables.items():
            if str(key).strip().lower() == region_name.strip().lower() and isinstance(value, pd.DataFrame):
                return value.copy()

    combined = get_dataframe(namespace, "combined_forecast_table")
    if combined is not None:
        for col in ["Region", "Country"]:
            if col in combined.columns:
                out = combined[combined[col].astype(str).str.lower().eq(region_name.lower())].copy()
                if not out.empty:
                    return out

    return None


def render_region_table(namespace: dict[str, Any], region_name: str) -> None:
    st.markdown(f"### Forecast table: {region_name}")

    table = find_region_table(namespace, region_name)

    if table is not None and not table.empty:
        show_dataframe(table)
        render_download_button_for_dataframe(
            table,
            f"forecast_table_{region_name.lower()}.csv",
            f"Download {region_name} forecast table CSV",
        )
    else:
        st.info(f"No forecast table was found for {region_name}.")


def render_optional_dataframe(
    namespace: dict[str, Any],
    dataframe_name: str,
    title: str,
    filename: str,
) -> None:
    df = get_dataframe(namespace, dataframe_name)

    if df is None or df.empty:
        return

    st.markdown("---")
    st.markdown(f"### {title}")
    show_dataframe(df)
    render_download_button_for_dataframe(
        df,
        filename,
        f"Download {title} CSV",
    )


# =====================================================================
# Provider comparison outputs for Overview
# =====================================================================

def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    for col in df.columns:
        lower = str(col).lower()
        if all(part.lower() in lower for part in candidates[0].split()):
            return col

    return None


def render_provider_comparison_outputs(namespace: dict[str, Any]) -> None:
    """
    Render overview-level ECMWF vs NOAA outputs as two clean figures.

    These are generated from provider_comparison_summary instead of relying on
    captured figure order, which prevents Peru polygon figures from appearing in
    the Overview.
    """
    provider_summary = get_dataframe(namespace, "provider_comparison_summary")
    provider_difference = get_dataframe(namespace, "provider_difference_summary")

    if provider_summary is None or provider_summary.empty:
        st.markdown("---")
        st.info("No ECMWF vs NOAA provider-comparison summary was found in the pipeline output.")
        return

    import matplotlib.pyplot as plt
    import numpy as np

    region_col = first_existing_column(provider_summary, ["Region", "Country"])
    provider_col = first_existing_column(provider_summary, ["Provider", "Forecast source", "Source"])
    temp_col = first_existing_column(
        provider_summary,
        ["Mean temperature anomaly (°C)", "Mean temp anomaly °C", "Mean temperature anomaly"],
    )
    precip_col = first_existing_column(
        provider_summary,
        [
            "Mean precipitation anomaly (mm/month)",
            "Mean precip anomaly mm/month",
            "Mean precipitation anomaly",
        ],
    )

    if region_col is None or provider_col is None:
        st.markdown("---")
        st.markdown("### ECMWF vs NOAA provider comparison")
        st.info("The provider comparison table does not contain the expected region/provider columns.")
        show_dataframe(provider_summary)
        return

    def render_grouped_bar_chart(value_col: str, title: str, ylabel: str) -> None:
        pivot = provider_summary.pivot_table(
            index=region_col,
            columns=provider_col,
            values=value_col,
            aggfunc="mean",
        )

        fig, ax = plt.subplots(figsize=(10, 4.8))
        x = np.arange(len(pivot.index))
        providers = list(pivot.columns)
        width = 0.75 / max(len(providers), 1)

        for i, provider in enumerate(providers):
            offset = (i - (len(providers) - 1) / 2) * width
            ax.bar(
                x + offset,
                pivot[provider].values,
                width=width,
                label=str(provider),
                alpha=0.82,
            )

        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=30, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend()
        plt.tight_layout()

        st.markdown("---")
        st.markdown(f"### {title}")
        render_limited_width_matplotlib_figure(fig)
        st.caption("ECMWF vs NOAA comparison based on the final common forecast months.")
        plt.close(fig)

    if temp_col is not None:
        render_grouped_bar_chart(
            temp_col,
            "ECMWF vs NOAA mean temperature anomaly",
            "°C",
        )

    if precip_col is not None:
        render_grouped_bar_chart(
            precip_col,
            "ECMWF vs NOAA mean precipitation anomaly",
            "mm/month",
        )

    with st.expander("Provider-comparison tables", expanded=False):
        st.markdown("#### ECMWF vs NOAA provider-comparison summary")
        show_dataframe(provider_summary)
        render_download_button_for_dataframe(
            provider_summary,
            "provider_comparison_summary.csv",
            "Download ECMWF vs NOAA comparison CSV",
        )

        if provider_difference is not None and not provider_difference.empty:
            st.markdown("#### ECMWF minus NOAA provider-difference summary")
            show_dataframe(provider_difference)
            render_download_button_for_dataframe(
                provider_difference,
                "provider_difference_summary.csv",
                "Download ECMWF minus NOAA difference CSV",
            )


# =====================================================================
# Peru polygon map
# =====================================================================

def render_peru_polygon_map(namespace: dict[str, Any]) -> None:
    """
    Render a simple Peru polygon/subregion map directly in Streamlit.

    This does not require Cartopy. It uses the shapely polygon geometries created
    by legacy_workflow.py:
    - peru_subregion_polygons
    - optional peru_country_geom
    """
    polygons = namespace.get("peru_subregion_polygons")
    peru_boundary = namespace.get("peru_country_geom")

    if not isinstance(polygons, dict) or len(polygons) == 0:
        st.markdown("---")
        st.info("No Peru polygon geometries were found in the pipeline output.")
        return

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon

    def iter_polygons(geom: Any) -> list[Any]:
        if geom is None:
            return []

        geom_type = getattr(geom, "geom_type", "")

        if geom_type == "Polygon":
            return [geom]

        if geom_type == "MultiPolygon":
            return list(geom.geoms)

        return []

    fig, ax = plt.subplots(figsize=(7.5, 9))

    if peru_boundary is not None:
        for geom in iter_polygons(peru_boundary):
            try:
                x, y = geom.exterior.xy
                ax.plot(x, y, color="black", linewidth=1.2, alpha=0.8)
            except Exception:
                pass

    patches = []
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(polygons), 1)))

    for idx, (name, geom) in enumerate(polygons.items()):
        for poly in iter_polygons(geom):
            try:
                coords = np.asarray(poly.exterior.coords)
                patches.append(MplPolygon(coords, closed=True))

                centroid = poly.centroid
                ax.text(
                    centroid.x,
                    centroid.y,
                    str(name),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.75,
                    ),
                )
            except Exception:
                continue

    if patches:
        collection = PatchCollection(
            patches,
            facecolor=colors[: len(patches)],
            edgecolor="black",
            linewidth=1.0,
            alpha=0.65,
        )
        ax.add_collection(collection)

    all_bounds = []
    for geom in polygons.values():
        try:
            all_bounds.append(geom.bounds)
        except Exception:
            pass

    if all_bounds:
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)

        pad_x = max((maxx - minx) * 0.12, 0.5)
        pad_y = max((maxy - miny) * 0.12, 0.5)

        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

    ax.set_title("Peru polygon subregions used for forecast analysis")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    st.markdown("---")
    st.markdown("### Peru polygon subregion map")
    render_limited_width_matplotlib_figure(fig)
    st.caption("Polygon subregions used for the Peru polygon-based forecast analysis.")
    plt.close(fig)


# =====================================================================
# Tab renderers
# =====================================================================

def render_overview_tab(namespace: dict[str, Any], figures: list[Any]) -> None:
    st.subheader("Overview")

    st.markdown(
        """
        This dashboard combines historical ERA5 observations with ECMWF SEAS5 seasonal forecasts
        to summarize expected temperature and precipitation anomalies across the study regions.

        The workflow builds a historical Niño3.4-based El Niño context, then compares the current
        forecast months against historical anomaly patterns and region-specific forecast tables.
        """
    )

    st.markdown("---")
    st.markdown("### Final forecast-month summary")

    summary = get_dataframe(namespace, "forecast_last3_summary")

    if summary is not None and not summary.empty:
        show_dataframe(summary)
        render_download_button_for_dataframe(
            summary,
            "forecast_last3_summary.csv",
            "Download final forecast-month summary CSV",
        )
    else:
        st.info("No forecast_last3_summary DataFrame was found in the pipeline output.")

    st.markdown("---")
    render_workflow_figure(figures, 1, title="Niño3.4 historical context")

    render_provider_comparison_outputs(namespace)


def render_region_tab(
    namespace: dict[str, Any],
    figures: list[Any],
    region_name: str,
    figure_numbers: list[int] | None = None,
    final_figure: int | None = None,
) -> None:
    render_region_table(namespace, region_name)

    if region_name.lower() == "peru":
        render_optional_dataframe(
            namespace,
            "peru_polygon_forecast_table",
            "Peru polygon forecast table",
            "peru_polygon_forecast_table.csv",
        )
        render_optional_dataframe(
            namespace,
            "peru_polygon_summary",
            "Peru polygon final-3-month summary",
            "peru_polygon_summary.csv",
        )

    matched_figures = find_region_figures(figures, region_name)
    figures_to_show = matched_figures if matched_figures else (figure_numbers or [])

    if final_figure is not None:
        figures_to_show = figures_to_show + [final_figure]

    if region_name.lower() == "peru":
        peru_polygon_figures: list[int] = []
        peru_polygon_figures += find_figures_by_terms(figures, ["peru", "polygon"])
        peru_polygon_figures += find_figures_by_terms(figures, ["peru", "subregion"])
        peru_polygon_figures += find_figures_by_terms(figures, ["polygon", "forecast"])
        peru_polygon_figures += find_figures_by_terms(figures, ["polygon", "comparison"])
        figures_to_show = figures_to_show + peru_polygon_figures

    figures_to_show = unique_existing_figure_numbers(figures_to_show, figures)

    if not figures_to_show:
        st.info(f"No captured figures were found for {region_name}.")
    else:
        for number in figures_to_show:
            st.markdown("---")
            render_workflow_figure(figures, number)

    if region_name.lower() == "peru":
        render_peru_polygon_map(namespace)


# =====================================================================
# Streamlit app
# =====================================================================

st.set_page_config(
    page_title="El Nino climate dashboard",
    page_icon="🌦️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 0.6rem;
        border: 1px solid #e6e8eb;
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cached_pipeline_run(workflow_hash: str, force_refresh: bool):
    return run_pipeline(
        data_dir=DATA_DIR,
        force_refresh=force_refresh,
    )


def main() -> None:
    st.title("🌦️ El Nino climate forecast dashboard")
    st.caption(
        "Overview plus regional forecast tables and selected figures for Peru, "
        "Madagascar, Thailand, Kenya, and Switzerland."
    )

    st.sidebar.title("Controls")
    force_refresh = st.sidebar.toggle(
        "Force fresh CDS download",
        value=False,
        help="Leave this off for normal dashboard use. Turn it on only when you want to redownload data.",
    )
    show_logs = st.sidebar.toggle("Show pipeline log", value=False)

    with st.spinner("Running workflow or loading cached output..."):
        result = cached_pipeline_run(WORKFLOW_HASH, force_refresh=force_refresh)

    if not result.ok:
        st.error("The climate pipeline did not complete.")
        if result.stdout:
            with st.expander("Pipeline log before the error", expanded=False):
                st.code(result.stdout, language="text")
        st.code(result.error, language="text")
        st.stop()

    ns = result.namespace
    figures = result.figures

    observed_end = ns.get("observed_end")
    historical_start = infer_historical_start(ns)
    forecast_start = ns.get("forecast_start")
    forecast_end = ns.get("forecast_end")
    latest_val = ns.get("latest_val")
    latest_time = ns.get("latest_time")

    latest_nino_value = None
    if latest_val is not None and latest_time is not None:
        latest_nino_value = f"{float(latest_val):.2f} deg C ({format_month(latest_time)})"

    c1, c2, c3, c4, c5 = st.columns([1.15, 1.0, 1.25, 0.85, 1.10])

    with c1:
        compact_metric("Historical data", f"{historical_start} to {format_month(observed_end)}")
    with c2:
        compact_metric("Forecast source", get_forecast_label(ns))
    with c3:
        compact_metric("ECMWF forecast period", format_range(forecast_start, forecast_end))
    with c4:
        compact_metric("Baseline", ns.get("BASELINE_LABEL"))
    with c5:
        compact_metric("Latest Nino3.4", latest_nino_value)

    with st.expander("Data coverage details", expanded=False):
        fc_times = ns.get("fc_times")
        try:
            forecast_months = pd.DatetimeIndex(pd.to_datetime(fc_times)).strftime("%Y-%m").tolist()
        except Exception:
            forecast_months = []

        st.write(
            {
                "historical_start": historical_start,
                "historical_end": format_month(observed_end),
                "ecmwf_forecast_months": forecast_months,
                "forecast_start": format_month(forecast_start),
                "forecast_end": format_month(forecast_end),
                "captured_figures": len(figures),
            }
        )

    if show_logs:
        with st.expander("Pipeline log", expanded=False):
            st.code(result.stdout, language="text")

    (
        tab_overview,
        tab_peru,
        tab_madagascar,
        tab_thailand,
        tab_kenya,
        tab_switzerland,
    ) = st.tabs(["Overview", "Peru", "Madagascar", "Thailand", "Kenya", "Switzerland"])

    with tab_overview:
        render_overview_tab(ns, figures)

    with tab_peru:
        render_region_tab(
            ns,
            figures,
            "Peru",
            figure_numbers=[4, 9, 13, 20, 21],
            final_figure=19,
        )

    with tab_madagascar:
        render_region_tab(ns, figures, "Madagascar", figure_numbers=[5, 10, 14])

    with tab_thailand:
        render_region_tab(ns, figures, "Thailand", figure_numbers=[3, 8, 12])

    with tab_kenya:
        render_region_tab(ns, figures, "Kenya", figure_numbers=[2, 7, 11])

    with tab_switzerland:
        render_region_tab(ns, figures, "Switzerland", figure_numbers=[6, 15])

    st.sidebar.markdown("---")
    st.sidebar.caption("Overview plus five regional tabs are shown in this version.")


if __name__ == "__main__":
    main()
