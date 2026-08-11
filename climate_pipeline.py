"""
Self-contained climate pipeline for the Streamlit dashboard.

This version keeps the stable historical ERA5 request through the previous full
year, then tries to append current-year ERA5 monthly means month by month. If CDS
rejects a near-real-time ERA5T month, that month is skipped instead of crashing
the dashboard.
"""
from __future__ import annotations
import contextlib, io, re, traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_DATA_DIR_NAME = "data_download"
WORKFLOW_HASH = "daac1fb1d2c5_clean_deploy_v3"
LEGACY_WORKFLOW_FILE = Path(__file__).resolve().with_name("legacy_workflow.py")

def load_legacy_workflow_source() -> str:
    """Load the legacy workflow from a normal Python file.

    Keeping the workflow in legacy_workflow.py makes the dashboard wrapper
    readable while preserving the existing patch-and-exec behavior.
    """
    return LEGACY_WORKFLOW_FILE.read_text(encoding="utf-8")


@dataclass
class PipelineResult:
    namespace: Dict[str, Any]
    stdout: str
    figures: List[Any]
    displays: List[Any]
    error: Optional[str] = None
    @property
    def ok(self) -> bool:
        return self.error is None

def patch_legacy_source(source: str, data_dir: Path, force_refresh: bool) -> str:
    data_dir = data_dir.resolve()
    data_dir_text = str(data_dir).replace("\\", "/")
    source = re.sub(
        r'DATADIR\s*=\s*["\']\.\/data_download["\']',
        lambda match: f'DATADIR = r"{data_dir_text}"',
        source,
    )

    # Keep the main bulk ERA5 request stable. Current-year months are appended
    # below one by one with try/except so unavailable ERA5T months do not crash.
    source = re.sub(
        r"ERA5_END_YEAR\s*=\s*CURRENT_YEAR",
        "ERA5_END_YEAR = CURRENT_YEAR - 1",
        source,
    )

    current_year_helpers = r'''

# ============================================================
# DASHBOARD PATCH: optional current-year ERA5 monthly extension
# ============================================================
# The main ERA5 request stops at the previous full year for stability.
# This helper then tries current-year months one by one. Months not yet
# available through CDS/ERA5T are skipped rather than failing the dashboard.
INCLUDE_CURRENT_YEAR_ERA5_MONTHS = True
CURRENT_YEAR_ERA5_MONTHS_USED = []

def _dashboard_current_year_months_to_try():
    if not INCLUDE_CURRENT_YEAR_ERA5_MONTHS:
        return []
    if RUN_DATE.month <= 1:
        return []
    return [f"{m:02d}" for m in range(1, RUN_DATE.month)]

def _dashboard_standardize_time(ds):
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds

def _dashboard_concat_time(datasets):
    datasets = [_dashboard_standardize_time(d) for d in datasets if d is not None]
    if len(datasets) == 0:
        return None
    if len(datasets) == 1:
        return datasets[0]
    out = xr.concat(datasets, dim="time")
    try:
        out = out.sortby("time")
    except Exception:
        pass
    return out
'''
    source = source.replace(
        "# ============================================================\n# 4. DOWNLOAD AND LOAD ERA5 NINO3.4 SST",
        current_year_helpers + "\n# ============================================================\n# 4. DOWNLOAD AND LOAD ERA5 NINO3.4 SST",
    )

    nino_append = r'''

# DASHBOARD PATCH: append available current-year monthly Nino3.4 SST
if INCLUDE_CURRENT_YEAR_ERA5_MONTHS:
    _current_nino_datasets = []
    for _m in _dashboard_current_year_months_to_try():
        _target = os.path.join(DATADIR, f"ERA5_nino34_sst_{CURRENT_YEAR}_{_m}.nc")
        try:
            _file = retrieve_fresh_if_needed(
                "reanalysis-era5-single-levels-monthly-means",
                {
                    "data_format": "netcdf",
                    "product_type": "monthly_averaged_reanalysis",
                    "variable": "sea_surface_temperature",
                    "year": [str(CURRENT_YEAR)],
                    "month": [_m],
                    "time": "00:00",
                    "area": [5, -170, -5, -120],
                },
                _target,
                force=FORCE_NINO_DOWNLOAD,
            )
            _current_nino_datasets.append(safe_open_cds_dataset(_file, required_vars=["sst"]))
            if f"{CURRENT_YEAR}-{_m}" not in CURRENT_YEAR_ERA5_MONTHS_USED:
                CURRENT_YEAR_ERA5_MONTHS_USED.append(f"{CURRENT_YEAR}-{_m}")
        except Exception as _e:
            print(f"Skipping current-year Nino3.4 ERA5 month {CURRENT_YEAR}-{_m}: {_e}")
    if len(_current_nino_datasets) > 0:
        ds_nino = _dashboard_concat_time([ds_nino] + _current_nino_datasets)
        print("Current-year Nino3.4 ERA5 months appended:", CURRENT_YEAR_ERA5_MONTHS_USED)
'''
    source = source.replace(
        'ds_nino = safe_open_cds_dataset(nino_file, required_vars=["sst"])\nprint_dataset_time_range(ds_nino, "Niño3.4 SST dataset")',
        'ds_nino = safe_open_cds_dataset(nino_file, required_vars=["sst"])\n' + nino_append + '\nprint_dataset_time_range(ds_nino, "Niño3.4 SST dataset")',
    )

    region_append = r'''

    # DASHBOARD PATCH: append available current-year monthly ERA5 data per region
    if INCLUDE_CURRENT_YEAR_ERA5_MONTHS:
        _current_region_datasets = []
        for _m in _dashboard_current_year_months_to_try():
            _target = os.path.join(DATADIR, f"ERA5_{region_name}_{CURRENT_YEAR}_{_m}_t2m_tp.nc")
            try:
                _file = retrieve_fresh_if_needed(
                    "reanalysis-era5-single-levels-monthly-means",
                    {
                        "data_format": "netcdf",
                        "product_type": "monthly_averaged_reanalysis",
                        "variable": ["2m_temperature", "total_precipitation"],
                        "year": [str(CURRENT_YEAR)],
                        "month": [_m],
                        "time": "00:00",
                        "area": regions[region_name],
                    },
                    _target,
                    force=FORCE_ERA5_DOWNLOAD,
                )
                _current_region_datasets.append(safe_open_cds_dataset(_file, required_vars=["t2m", "tp"]))
                if f"{CURRENT_YEAR}-{_m}" not in CURRENT_YEAR_ERA5_MONTHS_USED:
                    CURRENT_YEAR_ERA5_MONTHS_USED.append(f"{CURRENT_YEAR}-{_m}")
            except Exception as _e:
                print(f"Skipping current-year regional ERA5 month {region_name} {CURRENT_YEAR}-{_m}: {_e}")
        if len(_current_region_datasets) > 0:
            ds_reg = _dashboard_concat_time([ds_reg] + _current_region_datasets)
            print(f"Current-year ERA5 months appended for {region_name}:", sorted(CURRENT_YEAR_ERA5_MONTHS_USED))
'''
    source = source.replace(
        '    ds_reg = safe_open_cds_dataset(region_file, required_vars=["t2m", "tp"])\n    era5_region_data[region_name] = ds_reg',
        '    ds_reg = safe_open_cds_dataset(region_file, required_vars=["t2m", "tp"])\n' + region_append + '\n    era5_region_data[region_name] = ds_reg',
    )

    source = re.sub(
        r"\n# ------------------------------------------------------------\n# 17J\.5 Plot time series by Peru subregion[\s\S]*$",
        "\n# Optional Peru time-series block skipped in dashboard refactor.\n",
        source,
    )



    # ============================================================
    # DASHBOARD PATCH: keep forecast time-series lines only on true
    # future forecast months
    # ============================================================
    # Problem fixed:
    # In Figures 2-5, ECMWF forecast months are plotted on a Jan-Dec
    # calendar x-axis. If the latest ERA5 observed month moves forward,
    # the forecast line can visually appear over months that are no longer
    # forecast months. This patch filters common_fc_time so only months
    # strictly after observed_end are plotted as forecast.
    forecast_month_filter_patch = r"""
        observed_end_month = pd.Timestamp(observed_end).to_period("M").to_timestamp()

        common_fc_time = pd.DatetimeIndex([
            pd.Timestamp(t).to_period("M").to_timestamp()
            for t in common_fc_time
            if pd.Timestamp(t).to_period("M").to_timestamp() > observed_end_month
        ])
"""

    source = source.replace(
        '        # If you want to force only Aug-Dec, uncomment this:\n        # common_fc_time = common_fc_time[common_fc_time >= pd.Timestamp("2026-08-01")]',
        '        # DASHBOARD PATCH: automatic observed/forecast split\n        # Keep only true future forecast months after the latest observed ERA5 month.'
        + forecast_month_filter_patch,
    )

    source = source.replace(
        '        # Optional if you really want to drop July and plot only Aug-Dec:\n        # common_fc_time = common_fc_time[common_fc_time >= pd.Timestamp("2026-08-01")]',
        '        # DASHBOARD PATCH: automatic observed/forecast split\n        # Keep only true future forecast months after the latest observed ERA5 month.'
        + forecast_month_filter_patch,
    )

    refresh_value = "True" if force_refresh else "False"
    for flag in [
        "FORCE_NINO_DOWNLOAD",
        "FORCE_ERA5_DOWNLOAD",
        "FORCE_SEASONAL_DOWNLOAD",
        "FORCE_HINDCAST_YEAR_DISCOVERY",
    ]:
        source = re.sub(rf"{flag}\s*=\s*(True|False)", f"{flag} = {refresh_value}", source)
    return source

def run_pipeline(data_dir: str | Path = DEFAULT_DATA_DIR_NAME, force_refresh: bool = False) -> PipelineResult:
    data_path = Path(data_dir).resolve()
    data_path.mkdir(parents=True, exist_ok=True)
    captured_stdout = io.StringIO()
    figures: List[Any] = []
    displays: List[Any] = []
    original_show = plt.show
    def fake_show(*args: Any, **kwargs: Any) -> None:
        fig = plt.gcf()
        if fig is not None and fig.axes:
            figures.append(fig)
        plt.close(fig)
    def fake_display(obj: Any = None, *args: Any, **kwargs: Any) -> None:
        displays.append(obj)
    ns: Dict[str, Any] = {"__name__": "__dashboard_exec__", "display": fake_display}
    try:
        patched = patch_legacy_source(load_legacy_workflow_source(), data_path, force_refresh)
        plt.show = fake_show
        try:
            with contextlib.redirect_stdout(captured_stdout):
                exec(compile(patched, "embedded_legacy_workflow.py", "exec"), ns)
        finally:
            plt.show = original_show
        return PipelineResult(ns, captured_stdout.getvalue(), figures, displays, None)
    except Exception as exc:
        plt.show = original_show
        return PipelineResult(ns, captured_stdout.getvalue(), figures, displays, "".join(traceback.format_exception(exc)))

def get_dataframe(namespace: Dict[str, Any], name: str) -> Optional[pd.DataFrame]:
    obj = namespace.get(name)
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    return None

def get_all_dataframes(namespace: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    return {k: v.copy() for k, v in namespace.items() if isinstance(v, pd.DataFrame)}

def get_regions(namespace: Dict[str, Any]) -> List[str]:
    regions = namespace.get("regions", {})
    return list(regions.keys()) if isinstance(regions, dict) else []

def format_month(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m")
    except Exception:
        return "NA" if value is None else str(value)
