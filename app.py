"""
Streamlit app for the El Nino climate dashboard.

Tabs:
- Overview
- Peru
- Madagascar
- Thailand
- Kenya

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

def render_limited_width_matplotlib_figure(fig: Any, max_width_px: int = MAX_FIGURE_WIDTH_PX) -> None:
    """Render a matplotlib figure with a maximum display width.

    Streamlit's normal st.pyplot output expands with the page width, which makes
    figures too large on full-screen or large external monitors. Rendering the
    figure as an inline PNG inside a max-width HTML container keeps the figure
    readable and centered.
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
    <div style="max-width:{max_width_px}px; margin-left:auto; margin-right:auto;">
        <img src="data:image/png;base64,{encoded}"
             style="width:100%; height:auto; display:block;" />
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

st.set_page_config(
    page_title="El Nino climate dashboard",
    page_icon="🌦️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.15rem !important;
    }
    h1 {
        font-size: 2.10rem !important;
        line-height: 1.15 !important;
        margin-bottom: 0.15rem !important;
    }
    div[data-testid="stCaptionContainer"] {
        margin-bottom: 0.25rem !important;
    }
    .compact-metric {
        border: 1px solid #e6e8ef;
        border-radius: 0.55rem;
        background-color: white;
        padding: 0.42rem 0.58rem 0.48rem 0.58rem;
        min-height: 3.85rem;
    }
    .compact-metric-label {
        color: #5f6368;
        font-size: 0.70rem;
        line-height: 1.05;
        margin-bottom: 0.25rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .compact-metric-value {
        color: #31333f;
        font-size: 1.02rem;
        font-weight: 650;
        line-height: 1.12;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
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


def compact_metric(label: str, value: Any) -> None:
    value_text = "NA" if value is None else str(value)
    st.markdown(
        f"""
        <div class="compact-metric">
            <div class="compact-metric-label">{label}</div>
            <div class="compact-metric-value" title="{value_text}">{value_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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



def figure_matches_region(fig: Any, region_name: str) -> bool:
    """Return True if a captured matplotlib figure appears to belong to a region."""
    needle = region_name.lower()
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

    return any(needle in text.lower() for text in texts if text)


def find_region_figures(figures: list[Any], region_name: str) -> list[int]:
    """Find captured figure numbers whose titles/axis labels contain region_name."""
    matches: list[int] = []
    for i, fig in enumerate(figures, start=1):
        try:
            if figure_matches_region(fig, region_name):
                matches.append(i)
        except Exception:
            pass
    return matches

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


def render_numbered_figure(
    figures: list[Any],
    figure_number: int,
    title: str | None = None,
    show_missing_message: bool = False,
) -> None:
    index = figure_number - 1

    if not (0 <= index < len(figures)):
        if show_missing_message:
            label = title if title else f"Figure {figure_number}"
            st.markdown(f"### {label}")
            st.info(
                f"Figure {figure_number} was not captured. "
                f"The pipeline captured {len(figures)} figure(s)."
            )
        return

    label = title if title else f"Figure {figure_number}"
    st.markdown(f"### {label}")

    fig = fix_forecast_marker_and_legend(figures[index], figure_number)
    render_limited_width_matplotlib_figure(fig)
    st.caption(f"Showing Figure {figure_number} from the captured workflow output.")

def render_region_tab(
    namespace: dict[str, Any],
    figures: list[Any],
    region_name: str,
    figure_numbers: list[int] | None = None,
    final_figure: int | None = None,
) -> None:
    render_region_table(namespace, region_name)

    matched_figures = find_region_figures(figures, region_name)

    if matched_figures:
        figures_to_show = matched_figures
    else:
        figures_to_show = figure_numbers or []

    if final_figure is not None:
        figures_to_show = figures_to_show + [final_figure]

    # Keep the old requested numbers only if they actually exist.
    figures_to_show = [
        n for n in figures_to_show
        if 1 <= n <= len(figures)
    ]

    # Special Peru logic: explicitly include the Peru polygon figures
    # even if their captured figure numbers changed after deployment.
    if region_name.lower() == "peru":
        peru_polygon_figures = []

        peru_polygon_figures += find_figures_by_terms(
            figures,
            ["peru", "polygon", "forecast"]
        )

        peru_polygon_figures += find_figures_by_terms(
            figures,
            ["peru", "polygon", "comparison"]
        )

        peru_polygon_figures += find_figures_by_terms(
            figures,
            ["peru", "subregion"]
        )

        for n in peru_polygon_figures:
            if n not in figures_to_show and 1 <= n <= len(figures):
                figures_to_show.append(n)

    if not figures_to_show:
        st.info(f"No captured figures were found for {region_name}.")
        return

    for figure_number in figures_to_show:
        st.markdown("---")
        render_numbered_figure(figures, figure_number)


def render_overview_tab(namespace: dict[str, Any], figures: list[Any]) -> None:
    st.subheader("Overview")
    st.markdown(
        """
        This dashboard combines historical ERA5 observations with ECMWF SEAS5 seasonal forecasts to summarize expected temperature and precipitation anomalies across the four regions. The workflow builds a historical Niño3.4-based El Niño context, then compares the current forecast months against historical anomaly patterns and region-specific forecast tables.
        """
    )

    st.markdown("---")
    st.markdown("### Final forecast-month summary")
    summary = get_dataframe(namespace, "forecast_last3_summary")
    if summary is not None and not summary.empty:
        show_dataframe(summary)
        render_download_button_for_dataframe(summary, "forecast_last3_summary.csv", "Download final forecast-month summary CSV")
    else:
        st.info("No forecast_last3_summary DataFrame was found in the pipeline output.")

    st.markdown("---")
    render_numbered_figure(figures, 1)

    st.markdown("---")
    st.markdown("### Captured workflow figures")
    st.caption(
        f"The pipeline captured {len(figures)} figure(s). "
        "Only available figures are displayed, so missing legacy figure numbers are skipped."
    )

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


def figure_text_contains(fig: Any, required_terms: list[str]) -> bool:
    """
    Return True if all required terms appear somewhere in the figure titles,
    suptitle, axis labels, or legend labels.
    """
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
            handles, labels = ax.get_legend_handles_labels()
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


def main() -> None:
    st.title("🌦️ El Nino climate forecast dashboard")
    st.caption("Overview plus regional forecast tables and selected figures for Peru, Madagascar, Thailand, Kenya, and Switzerland.")

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

    tab_overview, tab_peru, tab_madagascar, tab_thailand, tab_kenya, tab_switzerland = st.tabs(["Overview", "Peru", "Madagascar", "Thailand", "Kenya", "Switzerland"])

    with tab_overview:
        render_overview_tab(ns, figures)
    with tab_peru:
        render_region_tab(ns, figures, "Peru", figure_numbers=[4, 9, 13, 20, 21], final_figure=19)
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
