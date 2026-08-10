# ============================================================
# What it does:
# 1. Downloads/loads latest ERA5 regional data and Niño3.4 SST
# 2. Finds latest common observed ERA5 month across regions
# 3. Finds latest ECMWF SEAS5 issue that has future forecast months
# 4. Uses all future forecast months actually available
# 5. Saves ECMWF as the main forecast provider
# 6. Leaves NOAA/NCEP CFSv2 available for final provider comparison only
# ============================================================

import os
import glob
import zipfile
import shutil
import warnings
import gc
from calendar import monthrange
from datetime import datetime

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import urllib3

urllib3.disable_warnings()
warnings.filterwarnings("ignore")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except Exception:
    CARTOPY_AVAILABLE = False
    print("Cartopy not available. Map plotting will be skipped.")


# ============================================================
# 1. SETTINGS
# ============================================================

DATADIR = "./data_download"
os.makedirs(DATADIR, exist_ok=True)
print("Data directory:", os.path.abspath(DATADIR))

c = cdsapi.Client()

regions = {
    "Kenya": [6, 30, -6, 45],
    "Thailand": [25, 94, 0, 110],
    "Peru": [5, -84, -20, -66],
    "Madagascar": [-8, 40, -28, 54],
    "Switzerland": [48.0, 5.8, 45.7, 10.7],
}

# Fixed anomaly baseline for comparability
BASELINE_START = "1991-01-01"
BASELINE_END = "2020-12-31"
BASELINE_LABEL = "1991-2020"

threshold = 0.5
min_duration = 5
plot_horizon = 12

RUN_DATE = pd.Timestamp.today().normalize()
CURRENT_YEAR = RUN_DATE.year

# Dynamic observed-data request window
OBSERVED_HISTORY_YEARS = 70
ERA5_START_YEAR = CURRENT_YEAR - OBSERVED_HISTORY_YEARS + 1
ERA5_END_YEAR = CURRENT_YEAR


# ============================================================
# 1A. FORECAST PROVIDER SETTINGS
#
# Main workflow:
#   - ECMWF SEAS5 is used for all main El Niño analysis plots.
#   - NOAA/NCEP CFSv2 is loaded only at the end for provider comparison.
# ============================================================

PRIMARY_FORECAST_SOURCE = "ecmwf_seas5"
COMPARISON_FORECAST_SOURCE = "noaa_ncep_cfsv2"

FORECAST_CONFIGS = {
    "ecmwf_seas5": {
        "label": "ECMWF SEAS5",
        "short_label": "ECMWF",
        "originating_centre": "ecmwf",
        "system": "51",
        "file_prefix": "ECMWF_SEAS5",
    },
    "noaa_ncep_cfsv2": {
        "label": "NOAA/NCEP CFSv2",
        "short_label": "NOAA",
        "originating_centre": "ncep",
        "system": "2",
        "file_prefix": "NOAA_NCEP_CFSv2",
    },
}


def set_forecast_source(source):
    """
    Activate one forecast provider by setting the global variables used by
    the seasonal retrieval and processing functions.

    Use ECMWF for the main analysis.
    Use NOAA only later for the final provider-comparison table/chart.
    """

    global FORECAST_SOURCE
    global FCFG
    global FORECAST_LABEL
    global FORECAST_SHORT_LABEL
    global SEASONAL_ORIGINATING_CENTRE
    global SEASONAL_SYSTEM
    global SEASONAL_FILE_PREFIX

    if source not in FORECAST_CONFIGS:
        raise ValueError(f"Unknown forecast source: {source}")

    FORECAST_SOURCE = source
    FCFG = FORECAST_CONFIGS[source]

    FORECAST_LABEL = FCFG["label"]
    FORECAST_SHORT_LABEL = FCFG["short_label"]
    SEASONAL_ORIGINATING_CENTRE = FCFG["originating_centre"]
    SEASONAL_SYSTEM = FCFG["system"]
    SEASONAL_FILE_PREFIX = FCFG["file_prefix"]

    print("\nActive forecast source:", FORECAST_LABEL)
    print("Originating centre:", SEASONAL_ORIGINATING_CENTRE)
    print("System:", SEASONAL_SYSTEM)


# Activate ECMWF as the main provider at the start
set_forecast_source(PRIMARY_FORECAST_SOURCE)


# ============================================================
# 1B. SEASONAL FORECAST REQUEST SETTINGS
# ============================================================

SEASONAL_LEAD_MONTHS_REQUESTED = [str(i) for i in range(1, 7)]
SEASONAL_LOOKBACK_MONTHS = 18

HINDCAST_LOOKBACK_YEARS = 70
MIN_HINDCAST_YEARS = 10

# Combined spatial request area covering all study regions
# Format: [latN, lonW, latS, lonE]
SEASONAL_AREA = [48.0, -84, -28, 110]

# Small area used only for probing hindcast-year availability
HINDCAST_PROBE_AREA = [1, 30, 0, 31]

FORCE_NINO_DOWNLOAD = True
FORCE_ERA5_DOWNLOAD = True
FORCE_SEASONAL_DOWNLOAD = True
FORCE_HINDCAST_YEAR_DISCOVERY = False


# ============================================================
# 1C. GLOBAL PLACEHOLDERS
# ============================================================

observed_end = None

seasonal_issue_year = None
seasonal_issue_month = None
selected_issue = None
hindcast_years_used = None

forecast_ds = None
hindcast_ds = None

fc_times = None
fc_lead_values = None

forecast_start = None
forecast_end = None
forecast_period = None

ongoing_event_start = None
ongoing_event_end = None

ecmwf_result = None
noaa_result = None


print("\nRun date:", RUN_DATE.strftime("%Y-%m-%d"))
print("Primary forecast source:", FORECAST_CONFIGS[PRIMARY_FORECAST_SOURCE]["label"])
print("Comparison forecast source:", FORECAST_CONFIGS[COMPARISON_FORECAST_SOURCE]["label"])
print("Observed history request:", ERA5_START_YEAR, "to", ERA5_END_YEAR)
print("Anomaly baseline:", BASELINE_LABEL)
print("Main rule: ECMWF is used for all main plots; NOAA is used only for the final provider-comparison chart.")


# ============================================================
# 2. FILE HELPERS
# ============================================================

def close_possible_xarray_objects():
    for name in list(globals()):
        obj = globals().get(name)
        if isinstance(obj, (xr.Dataset, xr.DataArray)):
            try:
                obj.close()
            except Exception:
                pass
    gc.collect()


def timestamped_path(path):
    root, ext = os.path.splitext(path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{root}_fresh_{stamp}{ext}"


def delete_file_and_extracted_parts(filepath):
    close_possible_xarray_objects()

    candidates = [
        filepath,
        filepath + ".zip",
        filepath.replace(".zip", "") + "_extracted",
        filepath.replace(".nc", "_extracted"),
        filepath.replace(".nc", ".nc_extracted"),
        os.path.splitext(filepath)[0] + "_extracted",
        os.path.splitext(filepath)[0] + "_parts",
    ]

    all_deleted = True

    for p in candidates:
        try:
            if os.path.isfile(p):
                os.remove(p)
                print("Deleted file:", p)
            elif os.path.isdir(p):
                shutil.rmtree(p)
                print("Deleted folder:", p)
        except PermissionError:
            all_deleted = False
            print("Could not delete because file is locked:", p)
        except Exception as e:
            all_deleted = False
            print("Could not delete:", p, "|", repr(e))

    return all_deleted


def validate_payload(filepath, min_size=1024):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    size = os.path.getsize(filepath)

    if size < min_size:
        raise OSError(f"File {filepath} is too small ({size} bytes).")

    with open(filepath, "rb") as f:
        header = f.read(8)

    if header[:3] == b"CDF" or header.startswith(b"\x89HDF"):
        return "netcdf"

    if header[:2] == b"PK" and zipfile.is_zipfile(filepath):
        return "zip"

    if zipfile.is_zipfile(filepath):
        return "zip"

    if header.lstrip().startswith(b"<") or header.lstrip().startswith(b"{"):
        with open(filepath, "r", errors="ignore") as f:
            sample = f.read(200)

        raise OSError(
            f"{os.path.basename(filepath)} is not valid data. "
            f"Sample: {sample[:100]!r}"
        )

    return "unknown"


def retrieve_fresh_if_needed(dataset_name, request, target_file, force=False):
    actual_file = target_file

    if force:
        deleted_ok = delete_file_and_extracted_parts(target_file)

        if not deleted_ok and os.path.exists(target_file):
            actual_file = timestamped_path(target_file)
            print("Original file locked. Downloading fresh file to:")
            print(" ", actual_file)

    if not os.path.isfile(actual_file):
        print("\nDownloading from CDS:")
        print(" ", actual_file)
        c.retrieve(dataset_name, request, actual_file)
    else:
        print("\nUsing existing local file:")
        print(" ", actual_file)

    return actual_file


def safe_open_cds_dataset(filepath, required_vars=None):
    if required_vars is None:
        required_vars = []

    def standardize_time(ds):
        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        if "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        return ds

    def has_required(ds):
        return all(v in ds.data_vars for v in required_vars)

    file_type = validate_payload(filepath)

    if file_type == "netcdf":
        ds = xr.open_dataset(filepath, engine="netcdf4")
        ds = standardize_time(ds)

        if not required_vars or has_required(ds):
            return ds

        print("Direct NetCDF opened but missing required variables.")
        print("Found variables:", list(ds.data_vars))

    if file_type == "zip":
        zip_path = filepath
    elif file_type == "unknown" and zipfile.is_zipfile(filepath):
        zip_path = filepath
    else:
        raise OSError(f"Could not open {filepath}: unrecognized format.")

    extract_dir = os.path.splitext(zip_path)[0] + "_extracted"
    os.makedirs(extract_dir, exist_ok=True)

    nc_files = glob.glob(os.path.join(extract_dir, "*.nc"))

    if len(nc_files) == 0:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        nc_files = glob.glob(os.path.join(extract_dir, "*.nc"))

    if len(nc_files) == 0:
        raise FileNotFoundError(f"No NetCDF files found after extracting {zip_path}")

    print(f"\nOpening extracted files from {filepath}:")

    datasets = []

    for nc in nc_files:
        ds_part = xr.open_dataset(nc, engine="netcdf4")
        ds_part = standardize_time(ds_part)
        print(" ", os.path.basename(nc), list(ds_part.data_vars))
        datasets.append(ds_part)

    ds = xr.merge(datasets, compat="override", join="outer")
    ds = standardize_time(ds)

    if required_vars and not has_required(ds):
        raise KeyError(
            f"Missing required variables {required_vars}. Found {list(ds.data_vars)}"
        )

    return ds


def print_dataset_time_range(ds, label):
    print(f"\n{label}")
    print("  variables:", list(ds.data_vars))

    if "time" in ds.coords:
        print(
            "  time range:",
            str(ds["time"].min().values)[:10],
            "to",
            str(ds["time"].max().values)[:10],
        )
    else:
        print("  no time coordinate found")


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def get_lat_lon_names(x):
    lat_name = "latitude" if "latitude" in x.coords else "lat"
    lon_name = "longitude" if "longitude" in x.coords else "lon"
    return lat_name, lon_name


def get_lead_dim(da):
    for d in ["forecastMonth", "leadtime_month"]:
        if d in da.dims:
            return d
    return None


def month_start(x):
    return pd.Timestamp(x).to_period("M").to_timestamp()


def get_time_max_month(ds):
    if "time" not in ds.coords:
        raise ValueError("Dataset has no time coordinate.")
    return month_start(ds["time"].max().values)


def set_observed_end_from_era5(era5_region_data):
    latest_by_region = {}

    for region_name, ds in era5_region_data.items():
        latest_by_region[region_name] = get_time_max_month(ds)

    common_latest = min(latest_by_region.values())

    print("\nLatest ERA5 month by region:")

    for region_name, latest in latest_by_region.items():
        print(f"  {region_name}: {latest:%Y-%m}")

    print(f"Using observed ERA5 through: {common_latest:%Y-%m}")

    return common_latest


def get_dataset_lead_values(ds):
    for v in ds.data_vars:
        da = ds[v]
        lead_dim = get_lead_dim(da)

        if lead_dim is not None:
            if lead_dim in da.coords:
                vals = da[lead_dim].values
            else:
                vals = np.arange(1, da.sizes[lead_dim] + 1)

            return np.array([int(x) for x in vals])

    raise ValueError("No forecastMonth or leadtime_month dimension found.")


def infer_forecast_valid_times_from_forecast_ds(ds, issue_year, issue_month):
    first_var = list(ds.data_vars)[0]
    da = ds[first_var]

    lead_dim = get_lead_dim(da)

    if lead_dim is None:
        raise ValueError(f"No forecast lead dimension found. Dims are: {da.dims}")

    if lead_dim in da.coords:
        lead_vals = da[lead_dim].values
    else:
        lead_vals = np.arange(1, da.sizes[lead_dim] + 1)

    if "forecast_reference_time" in da.coords:
        ref_time = month_start(da["forecast_reference_time"].values.ravel()[0])
    else:
        ref_time = pd.Timestamp(f"{issue_year}-{issue_month}-01")

    valid_times = []
    lead_ints = []

    for lead in lead_vals:
        lead_int = int(lead)
        valid_time = ref_time + pd.DateOffset(months=lead_int - 1)
        valid_times.append(month_start(valid_time))
        lead_ints.append(lead_int)

    return np.array(lead_ints), pd.DatetimeIndex(valid_times)


def make_issue_candidates(reference_date, lookback_months=18):
    """
    Candidate seasonal forecast initialisation months from latest to older.
    """
    start = pd.Timestamp(reference_date).to_period("M").to_timestamp()
    return [start - pd.DateOffset(months=i) for i in range(lookback_months)]


def region_slice_xr(ds, latN, lonW, latS, lonE):
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    lat_name, lon_name = get_lat_lon_names(ds)

    ds = ds.sortby(lat_name).sortby(lon_name)

    if float(ds[lon_name].max()) > 180:
        lonW = lonW % 360
        lonE = lonE % 360

    lat_slice = slice(latS, latN)

    if lonW <= lonE:
        out = ds.sel({lat_name: lat_slice, lon_name: slice(lonW, lonE)})
    else:
        out1 = ds.sel(
            {
                lat_name: lat_slice,
                lon_name: slice(lonW, float(ds[lon_name].max())),
            }
        )
        out2 = ds.sel(
            {
                lat_name: lat_slice,
                lon_name: slice(float(ds[lon_name].min()), lonE),
            }
        )
        out = xr.concat([out1, out2], dim=lon_name)

    return out


def area_weighted_mean(da):
    lat_name, lon_name = get_lat_lon_names(da)
    weights = np.cos(np.deg2rad(da[lat_name]))
    return da.weighted(weights).mean(dim=[lat_name, lon_name], skipna=True)


def clean_monthly_series(da):
    if "valid_time" in da.dims or "valid_time" in da.coords:
        da = da.rename({"valid_time": "time"})

    da = da.where(np.isfinite(da), drop=True)

    if "time" in da.coords:
        da = da.groupby("time").mean(skipna=True)

    return da


def clean_for_xarray_math(da):
    """
    Return a DataArray with only safe one-dimensional coordinates.

    This version uses the actual NumPy data shape, not da.sizes, because some
    CDS/xarray seasonal forecast arrays can retain stale coordinates after
    selection or arithmetic. That was causing errors like:

    CoordinateValidationError: conflicting sizes for dimension 'number':
    length 1 on the data but length 51 on coordinate 'number'
    """

    data = np.asarray(da.values)
    dims = tuple(da.dims)

    # If xarray metadata and actual data got out of sync, fall back to neutral
    # dimension names rather than carrying broken coordinates forward.
    if len(dims) != data.ndim:
        dims = tuple(f"dim_{i}" for i in range(data.ndim))

    coords = {}

    for axis, d in enumerate(dims):
        size = int(data.shape[axis])
        use_default = True

        if hasattr(da, "coords") and d in da.coords:
            coord = da.coords[d]
            vals = np.asarray(coord.values)

            # Keep only coordinates that exactly match the actual data shape.
            if coord.dims == (d,) and vals.ndim == 1 and vals.shape[0] == size:
                coords[d] = vals
                use_default = False

        if use_default:
            coords[d] = np.arange(size)

    return xr.DataArray(
        data=data,
        dims=dims,
        coords=coords,
        name=getattr(da, "name", None),
        attrs=getattr(da, "attrs", {}),
    )


def get_var(ds, candidates):
    for v in candidates:
        if v in ds.data_vars:
            return ds[v]

    raise KeyError(f"None of {candidates} found. Found {list(ds.data_vars)}")


def select_future_leads_and_rename_time(da):
    """
    Select the forecast leads chosen by set_dynamic_forecast_period, then rename
    the lead dimension to calendar time.

    This function is deliberately tolerant of CDS/xarray lead-coordinate quirks.
    If the DataArray exposes only positional lead values like [0] after previous
    cleaning, but the global forecast metadata has valid future fc_times, the
    function uses positional matching instead of crashing.
    """

    lead_dim = get_lead_dim(da)

    if lead_dim is None or lead_dim not in da.dims:
        return clean_for_xarray_math(da)

    if fc_times is None or fc_lead_values is None:
        raise ValueError(
            "fc_times/fc_lead_values not set. Run set_dynamic_forecast_period first."
        )

    data = np.asarray(da.values)
    lead_axis = list(da.dims).index(lead_dim)
    n_leads = int(data.shape[lead_axis])

    if n_leads == 0:
        return clean_for_xarray_math(da)

    if lead_dim in da.coords:
        raw_leads = np.asarray(da.coords[lead_dim].values).reshape(-1)
        if len(raw_leads) != n_leads:
            raw_leads = None
    else:
        raw_leads = None

    if raw_leads is None:
        raw_leads = np.arange(n_leads)

    lead_vals = []
    for value in raw_leads:
        try:
            lead_vals.append(int(value))
        except Exception:
            lead_vals.append(len(lead_vals))
    lead_vals = np.asarray(lead_vals, dtype=int)

    requested = [int(x) for x in np.asarray(fc_lead_values).reshape(-1)]
    requested_set = set(requested)

    keep_idx = [i for i, lead in enumerate(lead_vals) if int(lead) in requested_set]

    use_positional_matching = False

    if len(keep_idx) == 0:
        # Common problem after coordinate cleaning: lead coordinate becomes [0]
        # while fc_times/fc_lead_values still describe the correct forecast months.
        # In that case, keep the available lead positions and map them to fc_times.
        if n_leads <= len(fc_times):
            print(
                "Lead coordinate mismatch. Using positional lead/time matching. "
                f"Data leads={lead_vals.tolist()}, requested future leads={requested}"
            )
            keep_idx = list(range(n_leads))
            use_positional_matching = True
        else:
            raise ValueError(
                f"No matching future leads found. Data leads={lead_vals.tolist()}, "
                f"requested future leads={requested}"
            )

    kept_times = []

    for output_position, source_position in enumerate(keep_idx):
        if use_positional_matching:
            kept_times.append(fc_times[output_position])
            continue

        lead = int(lead_vals[source_position])
        if lead in requested_set:
            idx = requested.index(lead)
            kept_times.append(fc_times[idx])
        elif output_position < len(fc_times):
            kept_times.append(fc_times[output_position])
        else:
            kept_times.append(pd.Timestamp(fc_times[0]) + pd.DateOffset(months=output_position))

    da = da.isel({lead_dim: keep_idx})
    da = clean_for_xarray_math(da)

    if lead_dim != "time":
        da = da.rename({lead_dim: "time"})

    da = da.assign_coords(time=pd.DatetimeIndex(kept_times))

    return da

def detect_el_nino_events(oni_series, threshold=0.5, min_duration=5):
    oni_series = oni_series.dropna()
    is_event = oni_series >= threshold

    events = []
    current_start = None
    current_end = None

    for date, flag in is_event.items():
        if flag:
            if current_start is None:
                current_start = date
            current_end = date
        else:
            if current_start is not None:
                months = (
                    (current_end.year - current_start.year) * 12
                    + current_end.month
                    - current_start.month
                    + 1
                )

                if months >= min_duration:
                    events.append((current_start, current_end))

                current_start = None

    if current_start is not None:
        months = (
            (current_end.year - current_start.year) * 12
            + current_end.month
            - current_start.month
            + 1
        )

        if months >= min_duration:
            events.append((current_start, current_end))

    return events


def ensemble_quantiles(ens_da):
    if ens_da is None or ens_da.sizes.get("time", 0) == 0:
        return None

    if "number" not in ens_da.dims:
        raise ValueError(
            f"No ensemble member dimension called number. Dims are {ens_da.dims}"
        )

    q = ens_da.quantile(
        [0.05, 0.25, 0.50, 0.75, 0.95],
        dim="number",
        skipna=True
    )

    return {
        "p05": q.sel(quantile=0.05).values,
        "p25": q.sel(quantile=0.25).values,
        "median": q.sel(quantile=0.50).values,
        "p75": q.sel(quantile=0.75).values,
        "p95": q.sel(quantile=0.95).values,
        "time": ens_da.time.values,
    }


# ============================================================
# 4. DOWNLOAD AND LOAD ERA5 NINO3.4 SST
# ============================================================

nino_file_base = os.path.join(DATADIR, "ERA5_nino34_sst.nc")

nino_file = retrieve_fresh_if_needed(
    "reanalysis-era5-single-levels-monthly-means",
    {
        "data_format": "netcdf",
        "product_type": "monthly_averaged_reanalysis",
        "variable": "sea_surface_temperature",
        "year": [str(y) for y in range(ERA5_START_YEAR, ERA5_END_YEAR + 1)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        "area": [5, -170, -5, -120],
    },
    nino_file_base,
    force=FORCE_NINO_DOWNLOAD,
)

ds_nino = safe_open_cds_dataset(nino_file, required_vars=["sst"])
print_dataset_time_range(ds_nino, "Niño3.4 SST dataset")


# ============================================================
# 5. DOWNLOAD AND LOAD ERA5 REGIONAL DATA
# ============================================================

era5_region_files = {}

for region_name, area in regions.items():
    region_file_base = os.path.join(DATADIR, f"ERA5_{region_name}_t2m_tp.nc")

    actual_file = retrieve_fresh_if_needed(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "data_format": "netcdf",
            "product_type": "monthly_averaged_reanalysis",
            "variable": ["2m_temperature", "total_precipitation"],
            "year": [str(y) for y in range(ERA5_START_YEAR, ERA5_END_YEAR + 1)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": "00:00",
            "area": area,
        },
        region_file_base,
        force=FORCE_ERA5_DOWNLOAD,
    )

    era5_region_files[region_name] = actual_file


era5_region_data = {}

for region_name, region_file in era5_region_files.items():
    ds_reg = safe_open_cds_dataset(region_file, required_vars=["t2m", "tp"])
    era5_region_data[region_name] = ds_reg

    print_dataset_time_range(ds_reg, f"{region_name} ERA5 regional dataset")
    print("  t2m shape:", ds_reg["t2m"].shape)
    print("  tp shape:", ds_reg["tp"].shape)


observed_end = set_observed_end_from_era5(era5_region_data)


# ============================================================
# 6. FIND AND LOAD LATEST SEASONAL ISSUE WITH FUTURE MONTHS
# ============================================================

def seasonal_request_common(issue_month, variables, area, lead_months):
    return {
        "data_format": "netcdf",
        "originating_centre": SEASONAL_ORIGINATING_CENTRE,
        "system": SEASONAL_SYSTEM,
        "variable": variables,
        "product_type": "monthly_mean",
        "month": issue_month,
        "leadtime_month": lead_months,
        "area": area,
    }


def retrieve_seasonal_probe_for_year(issue_month, hindcast_year):
    """
    Probe one hindcast year for the currently active provider/month.
    Used only to discover which hindcast years are available.
    """

    probe_file = os.path.join(
        DATADIR,
        f"probe_{SEASONAL_FILE_PREFIX}_{issue_month}_{hindcast_year}.nc"
    )

    if FORCE_HINDCAST_YEAR_DISCOVERY:
        delete_file_and_extracted_parts(probe_file)

    if os.path.isfile(probe_file):
        try:
            validate_payload(probe_file)
            return True
        except Exception:
            delete_file_and_extracted_parts(probe_file)

    request = seasonal_request_common(
        issue_month=issue_month,
        variables=["2m_temperature"],
        area=HINDCAST_PROBE_AREA,
        lead_months=["1"]
    )

    request["year"] = str(hindcast_year)

    try:
        c.retrieve(
            "seasonal-monthly-single-levels",
            request,
            probe_file
        )
        validate_payload(probe_file)
        return True

    except Exception:
        delete_file_and_extracted_parts(probe_file)
        return False


def discover_available_hindcast_years(issue_month):
    """
    Dynamically discover available hindcast years by probing candidate years
    relative to the current run date.

    No fixed hindcast period is assumed.
    """

    cache_file = os.path.join(
        DATADIR,
        f"hindcast_years_{SEASONAL_FILE_PREFIX}_{issue_month}.txt"
    )

    if os.path.isfile(cache_file) and not FORCE_HINDCAST_YEAR_DISCOVERY:
        with open(cache_file, "r") as f:
            years = [
                int(x.strip())
                for x in f.readlines()
                if x.strip().isdigit()
            ]

        if len(years) >= MIN_HINDCAST_YEARS:
            print(
                f"Using cached hindcast years for {FORECAST_LABEL}, month {issue_month}:",
                years[0],
                "to",
                years[-1],
                f"n={len(years)}"
            )
            return years

    candidate_years = list(
        range(
            CURRENT_YEAR - HINDCAST_LOOKBACK_YEARS,
            CURRENT_YEAR + 1
        )
    )

    available_years = []

    print(
        f"\nDiscovering hindcast years for {FORECAST_LABEL}, issue month {issue_month}"
    )

    for y in candidate_years:
        ok = retrieve_seasonal_probe_for_year(
            issue_month=issue_month,
            hindcast_year=y
        )

        if ok:
            available_years.append(y)
            print("  available:", y)

    available_years = sorted(list(set(available_years)))

    if len(available_years) < MIN_HINDCAST_YEARS:
        raise RuntimeError(
            f"Only found {len(available_years)} hindcast years for "
            f"{FORECAST_LABEL}, month {issue_month}. "
            f"Need at least {MIN_HINDCAST_YEARS}."
        )

    with open(cache_file, "w") as f:
        for y in available_years:
            f.write(f"{y}\n")

    print(
        f"Discovered hindcast years for {FORECAST_LABEL}, month {issue_month}:",
        available_years[0],
        "to",
        available_years[-1],
        f"n={len(available_years)}"
    )

    return available_years


def retrieve_seasonal_for_issue(issue_date):
    issue_year = str(issue_date.year)
    issue_month = f"{issue_date.month:02d}"

    forecast_file_base = os.path.join(
        DATADIR,
        f"{SEASONAL_FILE_PREFIX}_{issue_year}_{issue_month}_area48N_future_t2m_tp.nc"
    )

    hindcast_years = discover_available_hindcast_years(issue_month)

    hindcast_file_base = os.path.join(
        DATADIR,
        f"{SEASONAL_FILE_PREFIX}_area48N_hindcast_dynamic_{issue_month}_t2m_tp.nc"
    )

    forecast_request = seasonal_request_common(
        issue_month=issue_month,
        variables=["2m_temperature", "total_precipitation"],
        area=SEASONAL_AREA,
        lead_months=SEASONAL_LEAD_MONTHS_REQUESTED
    )

    forecast_request["year"] = issue_year

    hindcast_request = seasonal_request_common(
        issue_month=issue_month,
        variables=["2m_temperature", "total_precipitation"],
        area=SEASONAL_AREA,
        lead_months=SEASONAL_LEAD_MONTHS_REQUESTED
    )

    hindcast_request["year"] = [
        str(y)
        for y in hindcast_years
    ]

    print("\nRetrieving seasonal forecast source:", FORECAST_LABEL)
    print("Issue:", issue_year, issue_month)
    print("Centre/system:", SEASONAL_ORIGINATING_CENTRE, SEASONAL_SYSTEM)
    print(
        "Dynamic hindcast years:",
        hindcast_years[0],
        "to",
        hindcast_years[-1],
        f"n={len(hindcast_years)}"
    )

    forecast_file = retrieve_fresh_if_needed(
        "seasonal-monthly-single-levels",
        forecast_request,
        forecast_file_base,
        force=FORCE_SEASONAL_DOWNLOAD,
    )

    hindcast_file = retrieve_fresh_if_needed(
        "seasonal-monthly-single-levels",
        hindcast_request,
        hindcast_file_base,
        force=FORCE_SEASONAL_DOWNLOAD,
    )

    forecast_ds_tmp = safe_open_cds_dataset(forecast_file)
    hindcast_ds_tmp = safe_open_cds_dataset(hindcast_file)

    return issue_year, issue_month, forecast_ds_tmp, hindcast_ds_tmp, hindcast_years


def set_dynamic_forecast_period(
    forecast_ds,
    hindcast_ds,
    issue_year,
    issue_month,
    observed_end
):
    """
    Forecast dates come from forecast_ds.
    Hindcast is only checked for lead availability.
    """

    global fc_times
    global fc_lead_values
    global forecast_start
    global forecast_end
    global forecast_period
    global ongoing_event_end
    global seasonal_issue_year
    global seasonal_issue_month

    seasonal_issue_year = str(issue_year)
    seasonal_issue_month = f"{int(issue_month):02d}"

    forecast_leads, forecast_valid_times = infer_forecast_valid_times_from_forecast_ds(
        forecast_ds,
        seasonal_issue_year,
        seasonal_issue_month
    )

    hindcast_leads = get_dataset_lead_values(hindcast_ds)

    common_leads = np.array([
        x
        for x in forecast_leads
        if x in set(hindcast_leads)
    ])

    keep_times = []
    keep_leads = []

    for lead, valid_time in zip(forecast_leads, forecast_valid_times):
        if lead in common_leads and valid_time > observed_end:
            keep_leads.append(lead)
            keep_times.append(valid_time)

    if len(keep_times) == 0:
        print("Issue:", seasonal_issue_year, seasonal_issue_month)
        print("Forecast leads:", forecast_leads.tolist())
        print("Forecast valid months:", forecast_valid_times.strftime("%Y-%m").tolist())
        print("Hindcast leads:", hindcast_leads.tolist())
        print("Observed end:", observed_end.strftime("%Y-%m"))

        raise ValueError(
            f"No future {FORECAST_LABEL} months found after observed_end."
        )

    fc_lead_values = np.array(keep_leads)
    fc_times = pd.DatetimeIndex(keep_times)

    forecast_start = fc_times[0]
    forecast_end = fc_times[-1]
    forecast_period = slice(forecast_start, forecast_end)
    ongoing_event_end = forecast_end

    print("\nSelected forecast source:", FORECAST_LABEL)
    print("Selected issue:", f"{seasonal_issue_year}-{seasonal_issue_month}")
    print("Forecast leads in file:", forecast_leads.tolist())
    print("Hindcast leads in file:", hindcast_leads.tolist())
    print("Forecast valid months in file:", forecast_valid_times.strftime("%Y-%m").tolist())
    print("Using future forecast months:", fc_times.strftime("%Y-%m").tolist())
    print("Using future lead values:", fc_lead_values.tolist())

    return fc_times


def load_latest_forecast_for_active_source():
    """
    Load the latest usable issue for the currently active forecast provider.

    This updates global forecast objects because the rest of the notebook
    expects forecast_ds, hindcast_ds, fc_times, forecast_period, etc.
    """

    global forecast_ds
    global hindcast_ds
    global selected_issue
    global hindcast_years_used

    global fc_times
    global fc_lead_values
    global forecast_start
    global forecast_end
    global forecast_period
    global ongoing_event_end
    global seasonal_issue_year
    global seasonal_issue_month

    forecast_ds = None
    hindcast_ds = None
    selected_issue = None
    hindcast_years_used = None

    fc_times = None
    fc_lead_values = None
    forecast_start = None
    forecast_end = None
    forecast_period = None
    ongoing_event_end = None
    seasonal_issue_year = None
    seasonal_issue_month = None

    issue_candidates = make_issue_candidates(
        RUN_DATE,
        SEASONAL_LOOKBACK_MONTHS
    )

    for issue_date in issue_candidates:
        print(f"\nTrying {FORECAST_LABEL} issue {issue_date:%Y-%m}")

        try:
            (
                issue_year,
                issue_month,
                forecast_ds_candidate,
                hindcast_ds_candidate,
                hindcast_years_candidate
            ) = retrieve_seasonal_for_issue(issue_date)

            set_dynamic_forecast_period(
                forecast_ds=forecast_ds_candidate,
                hindcast_ds=hindcast_ds_candidate,
                issue_year=issue_year,
                issue_month=issue_month,
                observed_end=observed_end,
            )

            forecast_ds = forecast_ds_candidate
            hindcast_ds = hindcast_ds_candidate
            selected_issue = pd.Timestamp(f"{issue_year}-{issue_month}-01")
            hindcast_years_used = hindcast_years_candidate

            print(f"Selected {FORECAST_LABEL} issue: {selected_issue:%Y-%m}")
            break

        except Exception as e:
            print(f"Skipping {FORECAST_LABEL} issue {issue_date:%Y-%m}: {repr(e)}")

    if forecast_ds is None or hindcast_ds is None:
        raise RuntimeError(
            f"No {FORECAST_LABEL} issue with future forecast months was found."
        )

    print("\nForecast source:", FORECAST_LABEL)
    print("Forecast variables:", list(forecast_ds.data_vars))
    print("Forecast dims:", forecast_ds.dims)

    for v in forecast_ds.data_vars:
        print(v, forecast_ds[v].dims, forecast_ds[v].shape)

    print("\nHindcast variables:", list(hindcast_ds.data_vars))
    print("Hindcast dims:", hindcast_ds.dims)

    for v in hindcast_ds.data_vars:
        print(v, hindcast_ds[v].dims, hindcast_ds[v].shape)

    print("Final forecast period:", f"{forecast_start:%Y-%m} to {forecast_end:%Y-%m}")
    print(
        "Dynamic hindcast years used:",
        hindcast_years_used[0],
        "to",
        hindcast_years_used[-1],
        f"n={len(hindcast_years_used)}"
    )

    print("\n================ DATA AVAILABILITY CHECK ================")
    print("Today / run date:", RUN_DATE.strftime("%Y-%m-%d"))
    print("Latest observed ERA5 month used:", observed_end.strftime("%Y-%m"))

    if fc_times is not None and len(fc_times) > 0:
        print("Forecast months currently used:", fc_times.strftime("%Y-%m").tolist())
    else:
        print("Forecast months currently used: none")

    print("=========================================================")

    return {
        "source": FORECAST_SOURCE,
        "label": FORECAST_LABEL,
        "short_label": FORECAST_SHORT_LABEL,
        "forecast_ds": forecast_ds,
        "hindcast_ds": hindcast_ds,
        "selected_issue": selected_issue,
        "hindcast_years_used": hindcast_years_used,
        "fc_times": fc_times,
        "fc_lead_values": fc_lead_values,
        "forecast_start": forecast_start,
        "forecast_end": forecast_end,
        "forecast_period": forecast_period,
        "seasonal_issue_year": seasonal_issue_year,
        "seasonal_issue_month": seasonal_issue_month,
    }


def activate_loaded_forecast(result):
    """
    Reactivate a previously loaded forecast result.

    This is important because the rest of the notebook uses global variables.
    """

    global forecast_ds
    global hindcast_ds
    global selected_issue
    global hindcast_years_used

    global fc_times
    global fc_lead_values
    global forecast_start
    global forecast_end
    global forecast_period
    global ongoing_event_end
    global seasonal_issue_year
    global seasonal_issue_month

    set_forecast_source(result["source"])

    forecast_ds = result["forecast_ds"]
    hindcast_ds = result["hindcast_ds"]
    selected_issue = result["selected_issue"]
    hindcast_years_used = result["hindcast_years_used"]

    fc_times = result["fc_times"]
    fc_lead_values = result["fc_lead_values"]
    forecast_start = result["forecast_start"]
    forecast_end = result["forecast_end"]
    forecast_period = result["forecast_period"]

    seasonal_issue_year = result["seasonal_issue_year"]
    seasonal_issue_month = result["seasonal_issue_month"]
    ongoing_event_end = forecast_end

    print("\nReactivated forecast source:", FORECAST_LABEL)
    print("Issue:", selected_issue.strftime("%Y-%m"))
    print("Forecast months:", fc_times.strftime("%Y-%m").tolist())


# ============================================================
# 6B. LOAD ECMWF AS THE MAIN FORECAST PROVIDER
# ============================================================

set_forecast_source(PRIMARY_FORECAST_SOURCE)
ecmwf_result = load_latest_forecast_for_active_source()
activate_loaded_forecast(ecmwf_result)


print("Today / run date:", RUN_DATE.strftime("%Y-%m-%d"))
print("Latest observed ERA5 month used:", observed_end.strftime("%Y-%m"))

if fc_times is not None and len(fc_times) > 0:
    print("Forecast months currently used:", fc_times.strftime("%Y-%m").tolist())
else:
    print("Forecast months currently used: none")

print("=========================================================")
# ============================================================
# 7. ERA5 REGIONAL ANOMALIES
# Fixed 1991-2020 baseline
# ============================================================

# Safety: make sure ECMWF is active for the main analysis
activate_loaded_forecast(ecmwf_result)

era5_anom_data = {}

for region_name, ds in era5_region_data.items():

    ds = ds.copy()

    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # Temperature: Kelvin to Celsius
    ds["t2m"] = ds["t2m"] - 273.15
    ds["t2m"].attrs["units"] = "°C"

    # Precipitation:
    # ERA5 monthly-mean total precipitation is converted to mm/month.
    # This makes it comparable with forecast monthly precipitation anomalies.
    days_in_month = xr.DataArray(
        [pd.Timestamp(t).days_in_month for t in ds.time.values],
        coords={"time": ds.time},
        dims=["time"]
    )

    ds["tp"] = ds["tp"] * 1000.0 * days_in_month
    ds["tp"].attrs["units"] = "mm/month"

    # Fixed baseline climatology
    clim_data = ds.sel(
        time=slice(BASELINE_START, BASELINE_END)
    )

    baseline_years_available = sorted(
        set(pd.DatetimeIndex(pd.to_datetime(clim_data.time.values)).year)
    )

    if len(baseline_years_available) == 0:
        raise ValueError(
            f"No ERA5 baseline years available for {region_name} "
            f"within {BASELINE_START} to {BASELINE_END}."
        )

    print(
        f"\n{region_name} ERA5 anomaly baseline:",
        BASELINE_LABEL,
        "| available years:",
        baseline_years_available[0],
        "to",
        baseline_years_available[-1],
        f"n={len(baseline_years_available)}"
    )

    clim = (
        clim_data
        .groupby("time.month")
        .mean("time", skipna=True)
    )

    era5_anom_data[region_name] = ds.groupby("time.month") - clim

print("\nERA5 regional anomalies ready.")
print(f"ERA5 anomaly baseline fixed to {BASELINE_LABEL}.")
print("ERA5 precipitation anomalies are in mm/month.")


# ============================================================
# 8. NINO3.4 INDEX
# Fixed 1991-2020 baseline
# ============================================================

ds_nino = ds_nino.copy()

if "valid_time" in ds_nino.dims or "valid_time" in ds_nino.coords:
    ds_nino = ds_nino.rename({"valid_time": "time"})

ds_nino["sst"] = ds_nino["sst"] - 273.15
ds_nino["sst"].attrs["units"] = "°C"

sst = ds_nino["sst"]

lat_name, lon_name = get_lat_lon_names(sst)

weights = np.cos(np.deg2rad(sst[lat_name]))
weights = weights / weights.mean()

sst_mean = (
    sst
    .weighted(weights)
    .mean(dim=[lat_name, lon_name], skipna=True)
)

# Fixed baseline climatology
sst_clim_data = sst_mean.sel(
    time=slice(BASELINE_START, BASELINE_END)
)

baseline_years_available = sorted(
    set(pd.DatetimeIndex(pd.to_datetime(sst_clim_data.time.values)).year)
)

if len(baseline_years_available) == 0:
    raise ValueError(
        f"No Niño3.4 baseline years available within "
        f"{BASELINE_START} to {BASELINE_END}."
    )

print(
    "\nNiño3.4 anomaly baseline:",
    BASELINE_LABEL,
    "| available years:",
    baseline_years_available[0],
    "to",
    baseline_years_available[-1],
    f"n={len(baseline_years_available)}"
)

sst_clim = (
    sst_clim_data
    .groupby("time.month")
    .mean("time", skipna=True)
)

nino34_index = (
    sst_mean.groupby("time.month") - sst_clim
).rename("nino34")

nino34_running = nino34_index.rolling(time=3).mean()

latest = nino34_running.dropna("time")
latest_val = float(latest.isel(time=-1).values)
latest_time = pd.Timestamp(latest.isel(time=-1).time.values)

print(f"\nLatest Niño3.4 3-month mean: {latest_time:%Y-%m} = {latest_val:.2f} °C")
print(f"Niño3.4 baseline fixed to {BASELINE_LABEL}.")


# ============================================================
# 9. FORECAST ANOMALIES
# ============================================================

def get_forecast_map_anomalies(region_name, latN, lonW, latS, lonE):

    hind_reg = region_slice_xr(
        hindcast_ds,
        latN,
        lonW,
        latS,
        lonE
    )

    fc_reg = region_slice_xr(
        forecast_ds,
        latN,
        lonW,
        latS,
        lonE
    )

    hind_t2 = clean_for_xarray_math(
        get_var(hind_reg, ["t2m", "2t"])
    )

    hind_tp = clean_for_xarray_math(
        get_var(hind_reg, ["tprate", "tp"])
    )

    fc_t2 = clean_for_xarray_math(
        get_var(fc_reg, ["t2m", "2t"])
    )

    fc_tp = clean_for_xarray_math(
        get_var(fc_reg, ["tprate", "tp"])
    )

    lead_fc_t2 = get_lead_dim(fc_t2)
    lead_fc_tp = get_lead_dim(fc_tp)

    lat_t2, lon_t2 = get_lat_lon_names(fc_t2)
    lat_tp, lon_tp = get_lat_lon_names(fc_tp)

    keep_fc_t2 = [
        d for d in [lead_fc_t2, "number", lat_t2, lon_t2]
        if d in fc_t2.dims
    ]

    keep_fc_tp = [
        d for d in [lead_fc_tp, "number", lat_tp, lon_tp]
        if d in fc_tp.dims
    ]

    extra_fc_t2 = [
        d for d in fc_t2.dims
        if d not in keep_fc_t2
    ]

    extra_fc_tp = [
        d for d in fc_tp.dims
        if d not in keep_fc_tp
    ]

    if extra_fc_t2:
        fc_t2 = fc_t2.mean(
            dim=extra_fc_t2,
            skipna=True
        )

    if extra_fc_tp:
        fc_tp = fc_tp.mean(
            dim=extra_fc_tp,
            skipna=True
        )

    lead_h_t2 = get_lead_dim(hind_t2)
    lead_h_tp = get_lead_dim(hind_tp)

    lat_h_t2, lon_h_t2 = get_lat_lon_names(hind_t2)
    lat_h_tp, lon_h_tp = get_lat_lon_names(hind_tp)

    keep_h_t2 = [
        d for d in [lead_h_t2, lat_h_t2, lon_h_t2]
        if d in hind_t2.dims
    ]

    keep_h_tp = [
        d for d in [lead_h_tp, lat_h_tp, lon_h_tp]
        if d in hind_tp.dims
    ]

    clim_dims_t2 = [
        d for d in hind_t2.dims
        if d not in keep_h_t2
    ]

    clim_dims_tp = [
        d for d in hind_tp.dims
        if d not in keep_h_tp
    ]

    hind_t2_clim = clean_for_xarray_math(
        hind_t2.mean(
            dim=clim_dims_t2,
            skipna=True
        )
    )

    hind_tp_clim = clean_for_xarray_math(
        hind_tp.mean(
            dim=clim_dims_tp,
            skipna=True
        )
    )

    # DASHBOARD PATCH: preserve forecast lead coordinates before anomaly math.
    # Small European regions exposed a bug where cleaned hindcast climatology
    # lead coordinates were [0] while forecast lead coordinates were [1..6].
    # xarray then aligned only the overlapping coordinates, collapsing or
    # emptying the forecast. Assigning the forecast lead coordinate to the
    # climatology keeps all forecast months available for tables and plots.
    if lead_fc_t2 in fc_t2.dims and lead_h_t2 in hind_t2_clim.dims:
        if fc_t2.sizes[lead_fc_t2] == hind_t2_clim.sizes[lead_h_t2]:
            if lead_h_t2 != lead_fc_t2:
                hind_t2_clim = hind_t2_clim.rename({lead_h_t2: lead_fc_t2})
            if lead_fc_t2 in fc_t2.coords:
                hind_t2_clim = hind_t2_clim.assign_coords(
                    {lead_fc_t2: fc_t2[lead_fc_t2].values}
                )

    if lead_fc_tp in fc_tp.dims and lead_h_tp in hind_tp_clim.dims:
        if fc_tp.sizes[lead_fc_tp] == hind_tp_clim.sizes[lead_h_tp]:
            if lead_h_tp != lead_fc_tp:
                hind_tp_clim = hind_tp_clim.rename({lead_h_tp: lead_fc_tp})
            if lead_fc_tp in fc_tp.coords:
                hind_tp_clim = hind_tp_clim.assign_coords(
                    {lead_fc_tp: fc_tp[lead_fc_tp].values}
                )

    t2_anom = clean_for_xarray_math(
        fc_t2 - hind_t2_clim
    )

    tp_anom = clean_for_xarray_math(
        fc_tp - hind_tp_clim
    )

    t2_anom = select_future_leads_and_rename_time(t2_anom)
    tp_anom = select_future_leads_and_rename_time(tp_anom)

    t2_anom.attrs["units"] = "°C anomaly"

    tp_name = fc_tp.name or ""

    if tp_name == "tprate":
        # m/s to mm/day
        tp_anom = tp_anom * 1000.0 * 86400.0
        tp_units = "mm/day"
    else:
        # m to mm
        tp_anom = tp_anom * 1000.0
        tp_units = "mm"

    tp_anom.attrs["units"] = tp_units

    return t2_anom, tp_anom, tp_units


def get_forecast_series_for_region(region_name, latN, lonW, latS, lonE):

    t2_map, tp_map, tp_units = get_forecast_map_anomalies(
        region_name,
        latN,
        lonW,
        latS,
        lonE
    )

    t2_series = clean_for_xarray_math(
        area_weighted_mean(t2_map)
    )

    tp_series = clean_for_xarray_math(
        area_weighted_mean(tp_map)
    )

    t2_series = t2_series.sel(time=forecast_period)
    tp_series = tp_series.sel(time=forecast_period)

    return t2_series, tp_series, tp_units


test_t2, test_tp, test_tp_units = get_forecast_series_for_region(
    "Kenya",
    *regions["Kenya"]
)

print("\nForecast test for Kenya:")
print("Forecast source:", FORECAST_LABEL)
print("Forecast months:", pd.to_datetime(test_t2.time.values).strftime("%Y-%m").tolist())
print("Forecast precip units before month conversion:", test_tp_units)


# ============================================================
# 10. FORECAST TABLES
# ============================================================

def build_forecast_tables_for_active_source():
    """
    Build forecast tables for the currently active forecast provider.
    Use ECMWF for the main analysis.
    Reuse this later for NOAA in the final provider-comparison block.
    """

    forecast_tables_out = {}

    for region_name, (latN, lonW, latS, lonE) in regions.items():

        t2_ens, tp_ens, tp_units = get_forecast_series_for_region(
            region_name,
            latN,
            lonW,
            latS,
            lonE
        )

        t2_med = (
            t2_ens.median(dim="number", skipna=True)
            if "number" in t2_ens.dims
            else t2_ens
        )

        tp_med = (
            tp_ens.median(dim="number", skipna=True)
            if "number" in tp_ens.dims
            else tp_ens
        )

        # Convert forecast precipitation from mm/day to mm/month if needed.
        if tp_units == "mm/day":

            days = xr.DataArray(
                [pd.Timestamp(t).days_in_month for t in tp_med.time.values],
                coords={"time": tp_med.time},
                dims=["time"]
            )

            tp_med = tp_med * days

        t2_med, tp_med = xr.align(
            t2_med,
            tp_med,
            join="inner"
        )

        month_dt = (
            pd.DatetimeIndex(pd.to_datetime(t2_med.time.values))
            .to_period("M")
            .to_timestamp()
        )

        df = pd.DataFrame({
            "Month": month_dt.strftime("%B %Y"),
            "Month_dt": month_dt,
            "Temperature anomaly (°C)": np.asarray(t2_med.values).ravel(),
            "Precipitation anomaly (mm/month)": np.asarray(tp_med.values).ravel(),
        })

        df["Temperature anomaly (°C)"] = df["Temperature anomaly (°C)"].round(2)
        df["Precipitation anomaly (mm/month)"] = df["Precipitation anomaly (mm/month)"].round(1)

        forecast_tables_out[region_name] = df

        print(f"\n{FORECAST_LABEL} - {region_name}")
        print("Months:", df["Month_dt"].dt.strftime("%Y-%m").tolist())
        display(df.drop(columns=["Month_dt"]))

    combined = pd.concat(
        [
            df.assign(
                Country=region_name,
                Provider=FORECAST_SHORT_LABEL
            )
            for region_name, df in forecast_tables_out.items()
        ],
        ignore_index=True
    )

    combined = combined[
        [
            "Provider",
            "Country",
            "Month",
            "Month_dt",
            "Temperature anomaly (°C)",
            "Precipitation anomaly (mm/month)"
        ]
    ]

    return forecast_tables_out, combined


# Main ECMWF forecast tables
activate_loaded_forecast(ecmwf_result)

ecmwf_forecast_tables, ecmwf_combined_forecast_table = build_forecast_tables_for_active_source()

# Keep old variable names so downstream code still works
forecast_tables = ecmwf_forecast_tables
combined_forecast_table = ecmwf_combined_forecast_table

display(combined_forecast_table.drop(columns=["Month_dt"]))


# ============================================================
# 11. EL NINO EVENT DETECTION AND 12-MONTH YEAR-CYCLE COMPARISON PLOTS
# ============================================================

# Safety: keep ECMWF active for this whole main-analysis section
activate_loaded_forecast(ecmwf_result)

plot_horizon = 12

month_labels = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

x_plot = np.arange(plot_horizon)


# ------------------------------------------------------------
# 11A. Detect El Niño events dynamically
# ------------------------------------------------------------

oni_series = nino34_running.to_series().dropna()
oni_series.index = pd.to_datetime(oni_series.index).to_period("M").to_timestamp()

el_nino_events = detect_el_nino_events(
    oni_series,
    threshold=threshold,
    min_duration=min_duration
)


def split_historical_and_current_events(events, observed_end, forecast_end=None):
    """
    Dynamically split detected El Niño events into:
      1. historical_events: past events used for the composite
      2. ongoing_event_start/end: latest current or recent event, if present

    Logic:
      - If the latest detected event ends close to the latest observed ERA5 month,
        treat it as the current/recent event and extend it with forecast months.
      - Otherwise, treat all events as historical and do not plot a current event.
    """

    if len(events) == 0:
        return [], None, None

    observed_end = pd.Timestamp(observed_end).to_period("M").to_timestamp()

    latest_start, latest_end = events[-1]
    latest_start = pd.Timestamp(latest_start).to_period("M").to_timestamp()
    latest_end = pd.Timestamp(latest_end).to_period("M").to_timestamp()

    months_since_latest_event = (
        (observed_end.year - latest_end.year) * 12
        + observed_end.month
        - latest_end.month
    )

    # If the latest detected event ended within 2 months of observed_end,
    # treat it as the current/recent event.
    if months_since_latest_event <= 2:
        historical_events = events[:-1]
        current_start = latest_start

        if forecast_end is not None:
            current_end = pd.Timestamp(forecast_end).to_period("M").to_timestamp()
        else:
            current_end = observed_end

    else:
        historical_events = events
        current_start = None
        current_end = None

    return historical_events, current_start, current_end


historical_events, ongoing_event_start, ongoing_event_end = split_historical_and_current_events(
    events=el_nino_events,
    observed_end=observed_end,
    forecast_end=forecast_end
)

print(f"\nDetected {len(el_nino_events)} formal El Niño events:")

for i, (start, end) in enumerate(el_nino_events, 1):

    duration = (
        (end.year - start.year) * 12
        + end.month
        - start.month
        + 1
    )

    print(
        f"  Event {i}: "
        f"{start:%Y-%m} to {end:%Y-%m} "
        f"({duration} months)"
    )

print(f"\nUsing {len(historical_events)} historical events for composite.")
print(f"Observed ERA5 ends: {observed_end:%Y-%m}")
print(f"{FORECAST_LABEL} forecast months used:", fc_times.strftime("%Y-%m").tolist())

if ongoing_event_start is not None:

    print(
        "Current/recent event detected dynamically:",
        f"{ongoing_event_start:%Y-%m} to {ongoing_event_end:%Y-%m}"
    )

else:

    print("No current/recent El Niño event detected dynamically.")


# ------------------------------------------------------------
# 11B. Quick ONI plot
# ------------------------------------------------------------

plt.figure(figsize=(11, 3.5))

plt.plot(
    oni_series.index,
    oni_series.values,
    color="black",
    linewidth=1.2
)

plt.axhline(
    threshold,
    color="red",
    linestyle="--",
    label=f"+{threshold:.1f}°C threshold"
)

for start, end in historical_events:

    plt.axvspan(
        start,
        end,
        color="red",
        alpha=0.18
    )

if ongoing_event_start is not None and ongoing_event_end is not None:

    plt.axvspan(
        ongoing_event_start,
        ongoing_event_end,
        color="purple",
        alpha=0.18,
        label="Current/recent event window"
    )

plt.title("Niño3.4 3-month running mean")
plt.ylabel("Niño3.4 anomaly, °C")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 11C. Prepare regional observed anomaly series
# ------------------------------------------------------------

region_anom_series = {}

for region_name in regions:

    ds_reg = era5_anom_data[region_name]

    if "valid_time" in ds_reg.dims or "valid_time" in ds_reg.coords:
        ds_reg = ds_reg.rename({"valid_time": "time"})

    t2m_series = clean_monthly_series(
        area_weighted_mean(ds_reg["t2m"])
    )

    tp_series = clean_monthly_series(
        area_weighted_mean(ds_reg["tp"])
    )

    t2m_series = t2m_series.sel(
        time=slice(None, observed_end)
    )

    tp_series = tp_series.sel(
        time=slice(None, observed_end)
    )

    region_anom_series[region_name] = {
        "t2m": t2m_series,
        "tp": tp_series
    }

print("\nObserved regional anomaly series ready.")

for region_name, series_dict in region_anom_series.items():

    tp_vals_check = np.asarray(
        series_dict["tp"].values,
        dtype=float
    )

    print(
        f"  {region_name}: "
        f"{pd.Timestamp(series_dict['t2m'].time.min().values):%Y-%m} to "
        f"{pd.Timestamp(series_dict['t2m'].time.max().values):%Y-%m}, "
        f"precip range = "
        f"{np.nanmin(tp_vals_check) if np.isfinite(tp_vals_check).any() else np.nan:.2f} to "
        f"{np.nanmax(tp_vals_check) if np.isfinite(tp_vals_check).any() else np.nan:.2f} mm/month"
    )


# ------------------------------------------------------------
# 11D. Build historical event-relative anomalies
# first 12 months only
# ------------------------------------------------------------

region_event_anoms = {region: [] for region in regions}

for start_date, end_date in historical_events:

    event_length = (
        (end_date.year - start_date.year) * 12
        + end_date.month
        - start_date.month
        + 1
    )

    rel_months = np.arange(event_length)

    for region_name, series_dict in region_anom_series.items():

        t_vals = series_dict["t2m"].sel(
            time=slice(start_date, end_date)
        ).values.astype(float)

        p_vals = series_dict["tp"].sel(
            time=slice(start_date, end_date)
        ).values.astype(float)

        if len(t_vals) < event_length:

            t_vals = np.pad(
                t_vals,
                (0, event_length - len(t_vals)),
                constant_values=np.nan
            )

        if len(p_vals) < event_length:

            p_vals = np.pad(
                p_vals,
                (0, event_length - len(p_vals)),
                constant_values=np.nan
            )

        region_event_anoms[region_name].append(
            (rel_months, t_vals, p_vals)
        )
# ============================================================
# 10B. SUMMARY TABLE FOR LAST 3 FORECAST MONTHS
# ECMWF main forecast summary
# ============================================================

# Safety: keep ECMWF active for main-analysis summaries
activate_loaded_forecast(ecmwf_result)

last_n_months = 3

summary_rows = []

for region_name, df in forecast_tables.items():

    df_tmp = df.copy()

    # Use existing Month_dt if available from updated Section 10.
    # If not, rebuild it from Month text as fallback.
    if "Month_dt" in df_tmp.columns:
        df_tmp["Month_dt"] = (
            pd.DatetimeIndex(pd.to_datetime(df_tmp["Month_dt"]))
            .to_period("M")
            .to_timestamp()
        )
    else:
        df_tmp["Month_dt"] = (
            pd.DatetimeIndex(
                pd.to_datetime(
                    df_tmp["Month"],
                    format="%B %Y"
                )
            )
            .to_period("M")
            .to_timestamp()
        )

    df_tmp = df_tmp.sort_values("Month_dt")

    # Take the final 3 available forecast months
    df_last3 = df_tmp.tail(last_n_months)

    summary_rows.append({
        "Provider": FORECAST_SHORT_LABEL,
        "Country": region_name,
        "Months used": ", ".join(df_last3["Month"].tolist()),
        "Mean temp anomaly °C": df_last3["Temperature anomaly (°C)"].mean(),
        "Max temp anomaly °C": df_last3["Temperature anomaly (°C)"].max(),
        "Mean precip anomaly mm/month": df_last3["Precipitation anomaly (mm/month)"].mean(),
        "Min precip anomaly mm/month": df_last3["Precipitation anomaly (mm/month)"].min(),
        "Max precip anomaly mm/month": df_last3["Precipitation anomaly (mm/month)"].max(),
    })


forecast_last3_summary = pd.DataFrame(summary_rows)

numeric_cols = [
    "Mean temp anomaly °C",
    "Max temp anomaly °C",
    "Mean precip anomaly mm/month",
    "Min precip anomaly mm/month",
    "Max precip anomaly mm/month"
]

forecast_last3_summary[numeric_cols] = forecast_last3_summary[numeric_cols].round(2)

print(f"\nSummary table for final {last_n_months} {FORECAST_LABEL} forecast months:")

display(
    forecast_last3_summary.style
    .hide(axis="index")
    .set_table_styles([
        {
            "selector": "th",
            "props": [
                ("background-color", "#1f4e79"),
                ("color", "white"),
                ("font-weight", "bold"),
                ("text-align", "center"),
                ("border", "1px solid white")
            ]
        },
        {
            "selector": "td",
            "props": [
                ("text-align", "right"),
                ("border", "1px solid white")
            ]
        },
        {
            "selector": "td:nth-child(1), td:nth-child(2), td:nth-child(3)",
            "props": [
                ("text-align", "left")
            ]
        }
    ])
    .set_properties(**{
        "font-size": "12pt",
        "padding": "6px"
    })
)

for region_name, df in forecast_tables.items():

    df_tmp = df.copy()

    if "Month_dt" in df_tmp.columns:
        df_tmp["Month_dt"] = (
            pd.DatetimeIndex(pd.to_datetime(df_tmp["Month_dt"]))
            .to_period("M")
            .to_timestamp()
        )
    else:
        df_tmp["Month_dt"] = (
            pd.DatetimeIndex(
                pd.to_datetime(
                    df_tmp["Month"],
                    format="%B %Y"
                )
            )
            .to_period("M")
            .to_timestamp()
        )

    df_tmp = df_tmp.sort_values("Month_dt")

    print(
        region_name,
        f"final {last_n_months} forecast months:",
        df_tmp.tail(last_n_months)["Month"].tolist()
    )
# ------------------------------------------------------------
# 11E. Build historical 12-month composites
# ------------------------------------------------------------

region_composite = {}

for region_name, events in region_event_anoms.items():

    if len(events) == 0:

        region_composite[region_name] = {
            "t2m_mean": np.full(plot_horizon, np.nan),
            "t2m_p25": np.full(plot_horizon, np.nan),
            "t2m_p75": np.full(plot_horizon, np.nan),
            "tp_mean": np.full(plot_horizon, np.nan),
            "tp_p25": np.full(plot_horizon, np.nan),
            "tp_p75": np.full(plot_horizon, np.nan),
        }

        print(f"{region_name}: no historical El Niño events available.")
        continue

    t_matrix = np.full(
        (len(events), plot_horizon),
        np.nan
    )

    p_matrix = np.full(
        (len(events), plot_horizon),
        np.nan
    )

    for i, (_, t_vals, p_vals) in enumerate(events):

        n_t = min(
            len(t_vals),
            plot_horizon
        )

        n_p = min(
            len(p_vals),
            plot_horizon
        )

        t_matrix[i, :n_t] = t_vals[:n_t]
        p_matrix[i, :n_p] = p_vals[:n_p]

    region_composite[region_name] = {
        "t2m_mean": np.nanmean(t_matrix, axis=0),
        "t2m_p25": np.nanpercentile(t_matrix, 25, axis=0),
        "t2m_p75": np.nanpercentile(t_matrix, 75, axis=0),
        "tp_mean": np.nanmean(p_matrix, axis=0),
        "tp_p25": np.nanpercentile(p_matrix, 25, axis=0),
        "tp_p75": np.nanpercentile(p_matrix, 75, axis=0),
    }

    comp_tp = region_composite[region_name]["tp_mean"]

    print(
        f"{region_name}: "
        f"{len(events)} historical El Niño events used, "
        f"historical precip composite range = "
        f"{np.nanmin(comp_tp) if np.isfinite(comp_tp).any() else np.nan:.2f} to "
        f"{np.nanmax(comp_tp) if np.isfinite(comp_tp).any() else np.nan:.2f} mm/month"
    )


# ------------------------------------------------------------
# 11F. Helper functions for plotting
# ------------------------------------------------------------

def convert_forecast_precip_for_plot(da, tp_units):
    """
    Convert forecast precipitation anomaly to mm/month if needed.

    Historical ERA5 precipitation is already mm/month from Section 7.
    If forecast precipitation is currently in mm/day, multiply by days/month.
    """

    da = da.copy()

    if tp_units == "mm/day":

        days = xr.DataArray(
            [pd.Timestamp(t).days_in_month for t in da.time.values],
            coords={"time": da.time},
            dims=["time"]
        )

        da = da * days

    return da


def standardize_da_monthly_time(da):
    """
    Force a DataArray time coordinate to monthly-start timestamps.
    This avoids mismatch between month-end, month-start, and valid_time formats.
    """

    da = da.copy()

    if "valid_time" in da.dims or "valid_time" in da.coords:
        da = da.rename({"valid_time": "time"})

    if "time" not in da.coords:
        raise ValueError("DataArray has no time coordinate.")

    month_time = (
        pd.DatetimeIndex(pd.to_datetime(da.time.values))
        .to_period("M")
        .to_timestamp()
    )

    da = da.assign_coords(time=month_time)

    # If duplicate monthly timestamps exist, average them
    da = da.groupby("time").mean(skipna=True)

    return da


def quantile_dict_to_dataframe(q):
    """
    Convert ensemble_quantiles output to a clean monthly dataframe.
    """

    if q is None:
        return pd.DataFrame(
            columns=["time", "p05", "p25", "median", "p75", "p95"]
        )

    q_time = (
        pd.DatetimeIndex(pd.to_datetime(q["time"]))
        .to_period("M")
        .to_timestamp()
    )

    df = pd.DataFrame({
        "time": q_time,
        "p05": np.asarray(q["p05"], dtype=float).ravel(),
        "p25": np.asarray(q["p25"], dtype=float).ravel(),
        "median": np.asarray(q["median"], dtype=float).ravel(),
        "p75": np.asarray(q["p75"], dtype=float).ravel(),
        "p95": np.asarray(q["p95"], dtype=float).ravel(),
    })

    df = (
        df
        .groupby("time", as_index=False)
        .mean(numeric_only=True)
        .sort_values("time")
        .reset_index(drop=True)
    )

    return df


def month_positions_from_time(time_values):
    """
    Convert timestamps to calendar-month positions.
    Jan=0, Feb=1, ..., Dec=11.
    """

    return np.array(
        [pd.Timestamp(t).month - 1 for t in time_values],
        dtype=int
    )


def filter_future_forecast_months(common_fc_time, observed_end):
    """
    Keep only forecast months strictly after the latest observed ERA5 month.
    This avoids plotting a forecast marker for a month that is already observed.
    """

    observed_end_month = pd.Timestamp(observed_end).to_period("M").to_timestamp()

    return pd.DatetimeIndex([
        pd.Timestamp(t).to_period("M").to_timestamp()
        for t in common_fc_time
        if pd.Timestamp(t).to_period("M").to_timestamp() > observed_end_month
    ])


def plot_discrete_calendar_forecast(ax, fc_idx, q_df, color, label_prefix):
    """
    Plot forecast months as discrete monthly markers with vertical uncertainty bars.

    This deliberately avoids a continuous forecast line and avoids fill_between.
    Therefore the forecast cannot visually extend through non-forecast months.
    """

    if q_df is None or len(q_df) == 0 or len(fc_idx) == 0:
        return

    x = np.asarray(fc_idx, dtype=float)

    med = np.asarray(q_df["median"].values, dtype=float)
    p05 = np.asarray(q_df["p05"].values, dtype=float)
    p25 = np.asarray(q_df["p25"].values, dtype=float)
    p75 = np.asarray(q_df["p75"].values, dtype=float)
    p95 = np.asarray(q_df["p95"].values, dtype=float)

    order = np.argsort(x)
    x = x[order]
    med = med[order]
    p05 = p05[order]
    p25 = p25[order]
    p75 = p75[order]
    p95 = p95[order]

    finite = (
        np.isfinite(x)
        & np.isfinite(med)
        & np.isfinite(p05)
        & np.isfinite(p25)
        & np.isfinite(p75)
        & np.isfinite(p95)
    )

    if finite.sum() == 0:
        return

    x = x[finite]
    med = med[finite]
    p05 = p05[finite]
    p25 = p25[finite]
    p75 = p75[finite]
    p95 = p95[finite]

    # Wide transparent vertical bars for the 5-95% range.
    ax.vlines(
        x,
        p05,
        p95,
        color=color,
        alpha=0.18,
        linewidth=8,
        label=f"{label_prefix} forecast 5-95%",
        zorder=4,
    )

    # Narrower vertical bars for the IQR.
    ax.vlines(
        x,
        p25,
        p75,
        color=color,
        alpha=0.45,
        linewidth=4,
        label=f"{label_prefix} forecast IQR",
        zorder=5,
    )

    # Median as individual monthly markers, not a continuous line.
    ax.scatter(
        x,
        med,
        marker="D",
        s=42,
        color=color,
        edgecolor="white",
        linewidth=0.7,
        label=f"{label_prefix} forecast median",
        zorder=6,
    )


# ------------------------------------------------------------
# 11G. ECMWF-only comparison plots
# ------------------------------------------------------------

# Safety: make sure the main plots use ECMWF, not NOAA
activate_loaded_forecast(ecmwf_result)

# Controls for forecast-period visual guide
SHOW_FORECAST_BOUNDARY_LINE = False
SHOW_FORECAST_BACKGROUND = True

print(
    "\nForecast months available for plotting:",
    pd.DatetimeIndex(pd.to_datetime(fc_times)).strftime("%Y-%m").tolist()
)

for region_name, (latN, lonW, latS, lonE) in regions.items():

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(15, 5),
        sharex=True
    )

    ax_t, ax_p = axes

    ax_t.set_title(
        f"{region_name} - temperature anomalies during El Niño"
    )

    ax_p.set_title(
        f"{region_name} - precipitation anomalies during El Niño"
    )

    ax_t.set_ylabel("Temperature anomaly, °C")
    ax_p.set_ylabel("Precipitation anomaly, mm/month")

    for ax in axes:

        ax.set_xlabel("Calendar month")

        ax.axhline(
            0,
            color="gray",
            linewidth=0.8
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.4
        )

        ax.set_xlim(
            0,
            plot_horizon - 1
        )

        ax.set_xticks(
            x_plot
        )

        ax.set_xticklabels(
            month_labels,
            rotation=0
        )

    # --------------------------------------------------------
    # Historical El Niño events
    # --------------------------------------------------------

    first_event = True

    n_events = len(region_event_anoms[region_name])

    label_event_indices = list(
        range(
            max(0, n_events - 3),
            n_events
        )
    )

    for event_idx, (rel_months, t_vals, p_vals) in enumerate(
        region_event_anoms[region_name]
    ):

        n = min(
            len(rel_months),
            len(t_vals),
            len(p_vals),
            plot_horizon
        )

        if n == 0:
            continue

        ax_t.plot(
            rel_months[:n],
            t_vals[:n],
            color="orange",
            linewidth=1.2,
            alpha=0.55,
            label="Historical events" if first_event else None
        )

        ax_p.plot(
            rel_months[:n],
            p_vals[:n],
            color="teal",
            linewidth=1.2,
            alpha=0.55,
            label="Historical events" if first_event else None
        )

        if event_idx in label_event_indices:

            event_year = historical_events[event_idx][0].year

            valid_t = np.where(
                np.isfinite(t_vals[:n])
            )[0]

            if len(valid_t) > 0:

                last_idx = valid_t[-1]

                ax_t.text(
                    rel_months[last_idx] + 0.10,
                    t_vals[last_idx],
                    str(event_year),
                    fontsize=8,
                    color="darkorange"
                )

            valid_p = np.where(
                np.isfinite(p_vals[:n])
            )[0]

            if len(valid_p) > 0:

                last_idx = valid_p[-1]

                ax_p.text(
                    rel_months[last_idx] + 0.10,
                    p_vals[last_idx],
                    str(event_year),
                    fontsize=8,
                    color="teal"
                )

        first_event = False

    # --------------------------------------------------------
    # Historical composite
    # --------------------------------------------------------

    comp = region_composite[region_name]

    ax_t.fill_between(
        x_plot,
        comp["t2m_p25"][:plot_horizon],
        comp["t2m_p75"][:plot_horizon],
        color="red",
        alpha=0.15,
        label="Historical IQR"
    )

    ax_t.plot(
        x_plot,
        comp["t2m_mean"][:plot_horizon],
        color="red",
        linewidth=2.5,
        label="Historical mean"
    )

    ax_p.fill_between(
        x_plot,
        comp["tp_p25"][:plot_horizon],
        comp["tp_p75"][:plot_horizon],
        color="blue",
        alpha=0.15,
        label="Historical IQR"
    )

    ax_p.plot(
        x_plot,
        comp["tp_mean"][:plot_horizon],
        color="blue",
        linewidth=2.5,
        label="Historical mean"
    )

    # --------------------------------------------------------
    # Current/recent observed event
    # --------------------------------------------------------

    if ongoing_event_start is not None:

        obs_t2 = clean_monthly_series(
            region_anom_series[region_name]["t2m"].sel(
                time=slice(
                    ongoing_event_start,
                    observed_end
                )
            )
        )

        obs_tp = clean_monthly_series(
            region_anom_series[region_name]["tp"].sel(
                time=slice(
                    ongoing_event_start,
                    observed_end
                )
            )
        )

        obs_t2 = standardize_da_monthly_time(obs_t2)
        obs_tp = standardize_da_monthly_time(obs_tp)

        obs_t2_time = pd.DatetimeIndex(pd.to_datetime(obs_t2.time.values))
        obs_tp_time = pd.DatetimeIndex(pd.to_datetime(obs_tp.time.values))

        common_obs_time = pd.DatetimeIndex(
            sorted(obs_t2_time.intersection(obs_tp_time))
        )

        if len(common_obs_time) > 0:

            obs_t2 = obs_t2.sel(time=common_obs_time)
            obs_tp = obs_tp.sel(time=common_obs_time)

            obs_t2_vals = obs_t2.values.astype(float)
            obs_tp_vals = obs_tp.values.astype(float)

            obs_idx = month_positions_from_time(common_obs_time)

        else:

            obs_t2_vals = np.array([], dtype=float)
            obs_tp_vals = np.array([], dtype=float)
            obs_idx = np.array([], dtype=int)

    else:

        common_obs_time = pd.DatetimeIndex([])
        obs_t2_vals = np.array([], dtype=float)
        obs_tp_vals = np.array([], dtype=float)
        obs_idx = np.array([], dtype=int)

    obs_len = len(obs_idx)

    if obs_len > 0:

        ax_t.plot(
            obs_idx,
            obs_t2_vals,
            color="black",
            linewidth=2.8,
            label="Current/recent event observed"
        )

        ax_p.plot(
            obs_idx,
            obs_tp_vals,
            color="black",
            linewidth=2.8,
            label="Current/recent event observed"
        )

    # --------------------------------------------------------
    # ECMWF forecast
    # --------------------------------------------------------

    forecast_months_found = []

    ens_t2_fc, ens_tp_fc, tp_units = get_forecast_series_for_region(
        region_name,
        latN,
        lonW,
        latS,
        lonE
    )

    ens_tp_fc = convert_forecast_precip_for_plot(
        ens_tp_fc,
        tp_units
    )

    q_t = ensemble_quantiles(ens_t2_fc)
    q_p = ensemble_quantiles(ens_tp_fc)

    q_t_df = quantile_dict_to_dataframe(q_t)
    q_p_df = quantile_dict_to_dataframe(q_p)

    if len(q_t_df) > 0 and len(q_p_df) > 0:

        common_fc_time = pd.DatetimeIndex(
            sorted(
                pd.DatetimeIndex(q_t_df["time"])
                .intersection(pd.DatetimeIndex(q_p_df["time"]))
            )
        )

        # DASHBOARD PATCH: automatic observed/forecast split
        # Keep only true future forecast months after the latest observed ERA5 month.
        common_fc_time = filter_future_forecast_months(
            common_fc_time,
            observed_end
        )

        q_t_df = (
            q_t_df
            .loc[q_t_df["time"].isin(common_fc_time)]
            .sort_values("time")
            .reset_index(drop=True)
        )

        q_p_df = (
            q_p_df
            .loc[q_p_df["time"].isin(common_fc_time)]
            .sort_values("time")
            .reset_index(drop=True)
        )

        if len(common_fc_time) > 0:

            fc_idx = month_positions_from_time(common_fc_time)

            forecast_months_found = common_fc_time.strftime("%Y-%m").tolist()

            plot_discrete_calendar_forecast(
                ax_t,
                fc_idx,
                q_t_df,
                color="firebrick",
                label_prefix=FORECAST_SHORT_LABEL
            )

            plot_discrete_calendar_forecast(
                ax_p,
                fc_idx,
                q_p_df,
                color="royalblue",
                label_prefix=FORECAST_SHORT_LABEL
            )

    # --------------------------------------------------------
    # Legend and validation output
    # --------------------------------------------------------

    for ax in axes:

        ax.legend(
            fontsize=8,
            loc="best"
        )

    plt.tight_layout()
    plt.show()

    print(f"\n{region_name}")
    print(f"  Historical events plotted: {len(region_event_anoms[region_name])}")
    print(
        "  Current observed months plotted:",
        [pd.Timestamp(t).strftime("%Y-%m") for t in common_obs_time]
    )
    print(f"  {FORECAST_LABEL} forecast months plotted: {forecast_months_found}")

    if obs_len > 0:

        print(
            f"  Observed precip range: "
            f"{np.nanmin(obs_tp_vals) if np.isfinite(obs_tp_vals).any() else np.nan:.2f} to "
            f"{np.nanmax(obs_tp_vals) if np.isfinite(obs_tp_vals).any() else np.nan:.2f} mm/month"
        )

    print(
        f"  Historical precip composite range: "
        f"{np.nanmin(comp['tp_mean']) if np.isfinite(comp['tp_mean']).any() else np.nan:.2f} to "
        f"{np.nanmax(comp['tp_mean']) if np.isfinite(comp['tp_mean']).any() else np.nan:.2f} mm/month"
    )

    # --------------------------------------------------------
    # Historical El Niño events
    # --------------------------------------------------------

    first_event = True

    n_events = len(region_event_anoms[region_name])

    label_event_indices = list(
        range(
            max(0, n_events - 3),
            n_events
        )
    )

    for event_idx, (rel_months, t_vals, p_vals) in enumerate(
        region_event_anoms[region_name]
    ):

        n = min(
            len(rel_months),
            len(t_vals),
            len(p_vals),
            plot_horizon
        )

        if n == 0:
            continue

        ax_t.plot(
            rel_months[:n],
            t_vals[:n],
            color="orange",
            linewidth=1.2,
            alpha=0.55,
            label="Historical events" if first_event else None
        )

        ax_p.plot(
            rel_months[:n],
            p_vals[:n],
            color="teal",
            linewidth=1.2,
            alpha=0.55,
            label="Historical events" if first_event else None
        )

        if event_idx in label_event_indices:

            event_year = historical_events[event_idx][0].year

            valid_t = np.where(
                np.isfinite(t_vals[:n])
            )[0]

            if len(valid_t) > 0:

                last_idx = valid_t[-1]

                ax_t.text(
                    rel_months[last_idx] + 0.10,
                    t_vals[last_idx],
                    str(event_year),
                    fontsize=8,
                    color="darkorange"
                )

            valid_p = np.where(
                np.isfinite(p_vals[:n])
            )[0]

            if len(valid_p) > 0:

                last_idx = valid_p[-1]

                ax_p.text(
                    rel_months[last_idx] + 0.10,
                    p_vals[last_idx],
                    str(event_year),
                    fontsize=8,
                    color="teal"
                )

        first_event = False

    # --------------------------------------------------------
    # Historical composite
    # --------------------------------------------------------

    comp = region_composite[region_name]

    ax_t.fill_between(
        x_plot,
        comp["t2m_p25"][:plot_horizon],
        comp["t2m_p75"][:plot_horizon],
        color="red",
        alpha=0.15,
        label="Historical IQR"
    )

    ax_t.plot(
        x_plot,
        comp["t2m_mean"][:plot_horizon],
        color="red",
        linewidth=2.5,
        label="Historical mean"
    )

    ax_p.fill_between(
        x_plot,
        comp["tp_p25"][:plot_horizon],
        comp["tp_p75"][:plot_horizon],
        color="blue",
        alpha=0.15,
        label="Historical IQR"
    )

    ax_p.plot(
        x_plot,
        comp["tp_mean"][:plot_horizon],
        color="blue",
        linewidth=2.5,
        label="Historical mean"
    )

    # --------------------------------------------------------
    # Current/recent observed event
    # --------------------------------------------------------

    if ongoing_event_start is not None:

        obs_t2 = clean_monthly_series(
            region_anom_series[region_name]["t2m"].sel(
                time=slice(
                    ongoing_event_start,
                    observed_end
                )
            )
        )

        obs_tp = clean_monthly_series(
            region_anom_series[region_name]["tp"].sel(
                time=slice(
                    ongoing_event_start,
                    observed_end
                )
            )
        )

        obs_t2 = standardize_da_monthly_time(obs_t2)
        obs_tp = standardize_da_monthly_time(obs_tp)

        obs_t2_time = pd.DatetimeIndex(pd.to_datetime(obs_t2.time.values))
        obs_tp_time = pd.DatetimeIndex(pd.to_datetime(obs_tp.time.values))

        common_obs_time = pd.DatetimeIndex(
            sorted(obs_t2_time.intersection(obs_tp_time))
        )

        if len(common_obs_time) > 0:

            obs_t2 = obs_t2.sel(time=common_obs_time)
            obs_tp = obs_tp.sel(time=common_obs_time)

            obs_t2_vals = obs_t2.values.astype(float)
            obs_tp_vals = obs_tp.values.astype(float)

            obs_idx = month_positions_from_time(common_obs_time)

        else:

            obs_t2_vals = np.array([], dtype=float)
            obs_tp_vals = np.array([], dtype=float)
            obs_idx = np.array([], dtype=int)

    else:

        common_obs_time = pd.DatetimeIndex([])
        obs_t2_vals = np.array([], dtype=float)
        obs_tp_vals = np.array([], dtype=float)
        obs_idx = np.array([], dtype=int)

    obs_len = len(obs_idx)

    if obs_len > 0:

        ax_t.plot(
            obs_idx,
            obs_t2_vals,
            color="black",
            linewidth=2.8,
            label="Current/recent event observed"
        )

        ax_p.plot(
            obs_idx,
            obs_tp_vals,
            color="black",
            linewidth=2.8,
            label="Current/recent event observed"
        )

    # --------------------------------------------------------
    # ECMWF forecast
    # --------------------------------------------------------

    forecast_months_found = []

    ens_t2_fc, ens_tp_fc, tp_units = get_forecast_series_for_region(
        region_name,
        latN,
        lonW,
        latS,
        lonE
    )

    ens_tp_fc = convert_forecast_precip_for_plot(
        ens_tp_fc,
        tp_units
    )

    q_t = ensemble_quantiles(ens_t2_fc)
    q_p = ensemble_quantiles(ens_tp_fc)

    q_t_df = quantile_dict_to_dataframe(q_t)
    q_p_df = quantile_dict_to_dataframe(q_p)

    if len(q_t_df) > 0 and len(q_p_df) > 0:

        common_fc_time = pd.DatetimeIndex(
            sorted(
                pd.DatetimeIndex(q_t_df["time"])
                .intersection(pd.DatetimeIndex(q_p_df["time"]))
            )
        )

        # DASHBOARD PATCH: automatic observed/forecast split
        # Keep only true future forecast months after the latest observed ERA5 month.
        common_fc_time = filter_future_forecast_months(
            common_fc_time,
            observed_end
        )

        q_t_df = (
            q_t_df
            .loc[q_t_df["time"].isin(common_fc_time)]
            .sort_values("time")
            .reset_index(drop=True)
        )

        q_p_df = (
            q_p_df
            .loc[q_p_df["time"].isin(common_fc_time)]
            .sort_values("time")
            .reset_index(drop=True)
        )

        if len(common_fc_time) > 0:

            fc_idx = month_positions_from_time(common_fc_time)

            forecast_months_found = common_fc_time.strftime("%Y-%m").tolist()

            plot_discrete_calendar_forecast(
                ax_t,
                fc_idx,
                q_t_df,
                color="firebrick",
                label_prefix=FORECAST_SHORT_LABEL
            )

            plot_discrete_calendar_forecast(
                ax_p,
                fc_idx,
                q_p_df,
                color="royalblue",
                label_prefix=FORECAST_SHORT_LABEL
            )

    # --------------------------------------------------------
    # Legend and validation output
    # --------------------------------------------------------

    for ax in axes:

        ax.legend(
            fontsize=8,
            loc="best"
        )

    plt.tight_layout()
    plt.show()

    print(f"\n{region_name}")
    print(f"  Historical events plotted: {len(region_event_anoms[region_name])}")
    print(
        "  Current observed months plotted:",
        [pd.Timestamp(t).strftime("%Y-%m") for t in common_obs_time]
    )
    print(f"  {FORECAST_LABEL} forecast months plotted: {forecast_months_found}")

    if obs_len > 0:

        print(
            f"  Observed precip range: "
            f"{np.nanmin(obs_tp_vals) if np.isfinite(obs_tp_vals).any() else np.nan:.2f} to "
            f"{np.nanmax(obs_tp_vals) if np.isfinite(obs_tp_vals).any() else np.nan:.2f} mm/month"
        )

    print(
        f"  Historical precip composite range: "
        f"{np.nanmin(comp['tp_mean']) if np.isfinite(comp['tp_mean']).any() else np.nan:.2f} to "
        f"{np.nanmax(comp['tp_mean']) if np.isfinite(comp['tp_mean']).any() else np.nan:.2f} mm/month"
    )
# ============================================================
# 12. FORECAST MAPS FOR FINAL 3 FORECAST MONTHS
# One figure per country, all 3 months plotted together
# Bottom colorbars
# ============================================================

# Safety: make sure maps use ECMWF, not NOAA
activate_loaded_forecast(ecmwf_result)


def sort_lat_lon(da):
    """
    Sort map data by latitude and longitude for clean plotting.
    """

    lat, lon = get_lat_lon_names(da)

    return da.sortby(lat).sortby(lon)




def coordinate_edges_for_map(values, lower_bound=None, upper_bound=None):
    """Convert 1D grid-cell centres to edges for pcolormesh."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.array([], dtype=float)

    values = np.sort(np.unique(values))

    if values.size == 1:
        step = 1.0
        edges = np.array([values[0] - step / 2.0, values[0] + step / 2.0])
    else:
        midpoints = (values[:-1] + values[1:]) / 2.0
        first_step = values[1] - values[0]
        last_step = values[-1] - values[-2]
        edges = np.concatenate([
            [values[0] - first_step / 2.0],
            midpoints,
            [values[-1] + last_step / 2.0],
        ])

    if lower_bound is not None:
        edges[0] = min(edges[0], float(lower_bound))
    if upper_bound is not None:
        edges[-1] = max(edges[-1], float(upper_bound))

    return edges


def plot_forecast_map_field(ax, da, levels, cmap, transform, extent):
    """
    Plot forecast anomaly maps.

    For regular/larger regions, this keeps contourf. For very small/coarse grids
    such as Switzerland, it uses pcolormesh with inferred grid-cell edges so the
    color field fills the map box instead of only the centre rectangle.
    """

    lat_name, lon_name = get_lat_lon_names(da)
    da = sort_lat_lon(da)

    n_lat = int(da.sizes[lat_name])
    n_lon = int(da.sizes[lon_name])

    if n_lat <= 4 or n_lon <= 4:
        lon_w, lon_e, lat_s, lat_n = extent
        lon_edges = coordinate_edges_for_map(
            da[lon_name].values,
            lower_bound=lon_w,
            upper_bound=lon_e,
        )
        lat_edges = coordinate_edges_for_map(
            da[lat_name].values,
            lower_bound=lat_s,
            upper_bound=lat_n,
        )

        cmap_obj = plt.get_cmap(cmap)
        norm = matplotlib.colors.BoundaryNorm(
            levels,
            ncolors=cmap_obj.N,
            clip=False,
        )

        return ax.pcolormesh(
            lon_edges,
            lat_edges,
            np.asarray(da.values, dtype=float),
            cmap=cmap_obj,
            norm=norm,
            shading="auto",
            transform=transform,
        )

    return ax.contourf(
        da[lon_name],
        da[lat_name],
        da,
        levels=levels,
        cmap=cmap,
        extend="both",
        transform=transform,
    )

def ensemble_median_map(da):
    """
    Reduce forecast map to ensemble median.
    If extra dimensions remain, average over them.
    """

    da = clean_for_xarray_math(da)

    if "number" in da.dims:

        da = da.median(
            dim="number",
            skipna=True
        )

    extra_dims = [
        d for d in da.dims
        if d not in [
            "time",
            "latitude",
            "longitude",
            "lat",
            "lon"
        ]
    ]

    if extra_dims:

        da = da.mean(
            dim=extra_dims,
            skipna=True
        )

    return da.squeeze(drop=True)


def select_fc_month(da, year, month):
    """
    Select a single forecast month from a DataArray.

    This is intentionally strict: it only returns a month that is actually
    present in the passed DataArray. The caller should first intersect requested
    months with available months for the region. This avoids silently plotting
    the wrong month when a regional forecast array has fewer valid months than
    the global fc_times index.
    """

    da = da.copy()

    if "valid_time" in da.dims or "valid_time" in da.coords:
        da = da.rename({"valid_time": "time"})

    if "time" not in da.coords:
        raise ValueError("DataArray has no time coordinate.")

    month_index = (
        pd.DatetimeIndex(pd.to_datetime(da.time.values))
        .to_period("M")
        .to_timestamp()
    )

    da = da.assign_coords(time=month_index)

    # Collapse duplicate monthly coordinates if they exist.
    da = da.groupby("time").mean(skipna=True)

    target_time = pd.Timestamp(f"{year}-{month:02d}-01")

    available_months = pd.DatetimeIndex(pd.to_datetime(da.time.values))

    if target_time not in available_months:
        raise KeyError(
            f"Forecast month {target_time:%Y-%m} is not available in this regional DataArray. "
            f"Available months: {available_months.strftime('%Y-%m').tolist()}"
        )

    out = da.sel(time=target_time)

    if "time" in out.dims:
        out = out.squeeze("time", drop=True)

    return out


def precip_month_total(da, year, month, units_kind):
    """
    Convert precipitation anomaly to mm/month if it is currently mm/day.
    """

    if units_kind == "mm/day":

        return da * monthrange(year, month)[1]

    return da


def get_final_forecast_months(n_months=3):
    """
    Return the final n available forecast months from fc_times.
    """

    if fc_times is None or len(fc_times) == 0:
        return pd.DatetimeIndex([])

    months = (
        pd.DatetimeIndex(pd.to_datetime(fc_times))
        .to_period("M")
        .to_timestamp()
    )

    months = pd.DatetimeIndex(
        sorted(months.unique())
    )

    return months[-n_months:]


def plot_country_forecast_month_grid(n_months=3):
    """
    Plot one figure per country.

    Each country figure has:
      - rows = final n forecast months
      - column 1 = temperature anomaly map
      - column 2 = precipitation anomaly map

    Colorbars are placed at the bottom in dedicated axes.
    """

    if not CARTOPY_AVAILABLE:

        print("Cartopy not available. Skipping map plots.")
        return

    target_months = get_final_forecast_months(n_months=n_months)

    if len(target_months) == 0:

        print("No forecast months available for map plotting.")
        return

    print(
        f"Using final {n_months} {FORECAST_LABEL} forecast months:",
        [m.strftime("%Y-%m") for m in target_months]
    )

    for region_name, (latN, lonW, latS, lonE) in regions.items():

        print(f"\nPreparing map grid for {region_name}")

        fc_t2_all, fc_tp_all, fc_tp_units = get_forecast_map_anomalies(
            region_name,
            latN,
            lonW,
            latS,
            lonE
        )

        # Use only months that are actually available for this region in both
        # temperature and precipitation arrays. The global fc_times may contain
        # months that a sliced regional DataArray does not expose after cleaning.
        fc_t2_months = pd.DatetimeIndex(pd.to_datetime(fc_t2_all.time.values)).to_period("M").to_timestamp()
        fc_tp_months = pd.DatetimeIndex(pd.to_datetime(fc_tp_all.time.values)).to_period("M").to_timestamp()

        available_region_months = pd.DatetimeIndex(
            sorted(set(fc_t2_months).intersection(set(fc_tp_months)))
        )

        region_target_months = pd.DatetimeIndex([
            m for m in target_months
            if pd.Timestamp(m).to_period("M").to_timestamp() in set(available_region_months)
        ])

        if len(region_target_months) == 0:
            print(
                f"Skipping map grid for {region_name}: no requested final forecast months are available. "
                f"Requested: {[m.strftime('%Y-%m') for m in target_months]}; "
                f"available: {available_region_months.strftime('%Y-%m').tolist()}"
            )
            continue

        if len(region_target_months) < len(target_months):
            print(
                f"Map grid for {region_name} uses available subset: "
                f"{region_target_months.strftime('%Y-%m').tolist()} "
                f"instead of requested {[m.strftime('%Y-%m') for m in target_months]}"
            )

        t2_maps = []
        tp_maps = []

        for month_date in region_target_months:

            t2m_map = ensemble_median_map(
                select_fc_month(
                    fc_t2_all,
                    month_date.year,
                    month_date.month
                )
            )

            tp_map = ensemble_median_map(
                select_fc_month(
                    fc_tp_all,
                    month_date.year,
                    month_date.month
                )
            )

            tp_map = precip_month_total(
                tp_map,
                month_date.year,
                month_date.month,
                fc_tp_units
            )

            t2m_map = sort_lat_lon(t2m_map)
            tp_map = sort_lat_lon(tp_map)

            t2_maps.append(t2m_map)
            tp_maps.append(tp_map)

        # ----------------------------------------------------
        # Shared color scales within each country figure
        # ----------------------------------------------------

        t_abs_values = []

        for t2m_map in t2_maps:

            t_abs_values.extend([
                abs(float(t2m_map.min(skipna=True))),
                abs(float(t2m_map.max(skipna=True)))
            ])

        t_abs = max(
            t_abs_values + [0.1]
        )

        t_levels = np.linspace(
            -t_abs,
            t_abs,
            21
        )

        p_abs_values = []

        for tp_map in tp_maps:

            p_abs_values.extend([
                abs(float(tp_map.min(skipna=True))),
                abs(float(tp_map.max(skipna=True)))
            ])

        p_abs = max(
            p_abs_values + [0.1]
        )

        p_levels = np.linspace(
            -p_abs,
            p_abs,
            21
        )

        # ----------------------------------------------------
        # Plot one country figure with all months together
        # ----------------------------------------------------

        proj = ccrs.PlateCarree()

        fig, axes = plt.subplots(
            nrows=len(region_target_months),
            ncols=2,
            figsize=(11.5, 4.0 * len(region_target_months)),
            subplot_kw={"projection": proj}
        )

        if len(region_target_months) == 1:
            axes = np.array([axes])

        last_im_t = None
        last_im_p = None

        for i, month_date in enumerate(region_target_months):

            t2m_map = t2_maps[i]
            tp_map = tp_maps[i]

            lat, lon = get_lat_lon_names(t2m_map)

            ax_t = axes[i, 0]
            ax_p = axes[i, 1]

            # --------------------------------------------
            # Temperature anomaly
            # --------------------------------------------

            last_im_t = plot_forecast_map_field(
                ax_t,
                t2m_map,
                levels=t_levels,
                cmap="coolwarm",
                transform=proj,
                extent=[lonW, lonE, latS, latN]
            )

            ax_t.set_title(
                f"{month_date:%B %Y} temperature anomaly",
                fontsize=12,
                pad=6
            )

            ax_t.coastlines(linewidth=0.7)

            ax_t.add_feature(
                cfeature.BORDERS,
                linewidth=0.5
            )

            ax_t.set_extent(
                [lonW, lonE, latS, latN],
                crs=proj
            )

            # --------------------------------------------
            # Precipitation anomaly
            # --------------------------------------------

            last_im_p = plot_forecast_map_field(
                ax_p,
                tp_map,
                levels=p_levels,
                cmap="BrBG",
                transform=proj,
                extent=[lonW, lonE, latS, latN]
            )

            ax_p.set_title(
                f"{month_date:%B %Y} precipitation anomaly",
                fontsize=12,
                pad=6
            )

            ax_p.coastlines(linewidth=0.7)

            ax_p.add_feature(
                cfeature.BORDERS,
                linewidth=0.5
            )

            ax_p.set_extent(
                [lonW, lonE, latS, latN],
                crs=proj
            )

            print(
                f"  {month_date:%Y-%m}: "
                f"temperature range = "
                f"{float(t2m_map.min(skipna=True)):.2f} to "
                f"{float(t2m_map.max(skipna=True)):.2f} °C, "
                f"precip range = "
                f"{float(tp_map.min(skipna=True)):.2f} to "
                f"{float(tp_map.max(skipna=True)):.2f} mm/month"
            )

        # ----------------------------------------------------
        # Layout and bottom colorbars
        # ----------------------------------------------------

        # Important:
        # Do not use tight_layout here, because it tends to pull the maps
        # down over the colorbars.
        fig.subplots_adjust(
            left=0.04,
            right=0.96,
            bottom=0.17,
            top=0.92,
            hspace=0.25,
            wspace=0.08
        )

        # Dedicated bottom colorbar axes.
        # Format: [left, bottom, width, height] in figure coordinates.
        cbar_t_ax = fig.add_axes(
            [0.09, 0.075, 0.36, 0.025]
        )

        cbar_p_ax = fig.add_axes(
            [0.55, 0.075, 0.36, 0.025]
        )

        cbar_t = fig.colorbar(
            last_im_t,
            cax=cbar_t_ax,
            orientation="horizontal"
        )

        cbar_t.set_label(
            "Temperature anomaly, °C",
            fontsize=10,
            labelpad=6
        )

        cbar_t.ax.tick_params(
            labelsize=9,
            pad=2
        )

        cbar_p = fig.colorbar(
            last_im_p,
            cax=cbar_p_ax,
            orientation="horizontal"
        )

        cbar_p.set_label(
            "Precipitation anomaly, mm/month",
            fontsize=10,
            labelpad=6
        )

        cbar_p.ax.tick_params(
            labelsize=9,
            pad=2
        )

        fig.suptitle(
            f"{region_name}: final {n_months} {FORECAST_LABEL} forecast months",
            y=0.975,
            fontsize=15
        )

        plt.show()


plot_country_forecast_month_grid(n_months=3)


# ============================================================
# 13. YEAR-ON-YEAR HEATMAPS + EL NINO FORECAST COMPARISON
#
# Figure 2:
#   Year-on-year heatmaps for all years, not El Niño filtered.
#   Includes latest observed year and latest forecast year dynamically.
#
# Figure 3:
#   Final 3 forecast months compared against historical El Niño years only.
#   This remains El Niño-specific.
# ============================================================

# Safety: make sure this section uses ECMWF, not NOAA
activate_loaded_forecast(ecmwf_result)


# ------------------------------------------------------------
# 13A. Basic settings
# ------------------------------------------------------------

month_labels = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

month_numbers = np.arange(1, 13)

if fc_times is not None and len(fc_times) > 0:
    forecast_years_available = sorted(
        list(dict.fromkeys([pd.Timestamp(t).year for t in fc_times]))
    )
else:
    forecast_years_available = []

latest_observed_year = pd.Timestamp(observed_end).year

if len(forecast_years_available) > 0:
    max_plot_year = max(latest_observed_year, max(forecast_years_available))
else:
    max_plot_year = latest_observed_year

print("\nPrep figures")
print("Observed ERA5 ends:", pd.Timestamp(observed_end).strftime("%Y-%m"))

if fc_times is not None and len(fc_times) > 0:
    print("Latest forecast months available:", fc_times.strftime("%Y-%m").tolist())
else:
    print("No forecast months available.")


# ------------------------------------------------------------
# 13B. Helper functions
# ------------------------------------------------------------

def standardize_monthly_time(da):
    """
    Force xarray time coordinate to monthly-start timestamps.
    This avoids silent reindexing failures.
    """

    da = da.copy()

    if "valid_time" in da.dims or "valid_time" in da.coords:
        da = da.rename({"valid_time": "time"})

    if "time" not in da.coords:
        raise ValueError("DataArray has no time coordinate.")

    new_time = (
        pd.DatetimeIndex(pd.to_datetime(da.time.values))
        .to_period("M")
        .to_timestamp()
    )

    da = da.assign_coords(time=new_time)
    da = da.groupby("time").mean(skipna=True)

    return da


def calendar_year_months(year):
    return pd.date_range(
        f"{year}-01-01",
        f"{year}-12-01",
        freq="MS"
    )


def event_label_from_dates(start, end):
    if start.year == end.year:
        return f"{start.year}"

    return f"{start.year}-{str(end.year)[-2:]}"


def convert_forecast_precip_for_plot(da, tp_units):
    """
    Convert forecast precipitation anomaly to mm/month if needed.

    If tp_units == "mm/day", multiply by number of days in each forecast month.
    """

    da = da.copy()

    if tp_units == "mm/day":

        days = xr.DataArray(
            [pd.Timestamp(t).days_in_month for t in da.time.values],
            coords={"time": da.time},
            dims=["time"]
        )

        da = da * days

    return da


def quantile_output_to_arrays(q):
    """
    Convert ensemble_quantiles output to clean numpy arrays.
    Time is standardized to monthly-start timestamps.
    """

    if q is None:

        return {
            "time": pd.DatetimeIndex([]),
            "x": np.array([], dtype=int),
            "year": np.array([], dtype=int),
            "p05": np.array([], dtype=float),
            "p25": np.array([], dtype=float),
            "p50": np.array([], dtype=float),
            "p75": np.array([], dtype=float),
            "p95": np.array([], dtype=float),
        }

    q_time = (
        pd.DatetimeIndex(pd.to_datetime(q["time"]))
        .to_period("M")
        .to_timestamp()
    )

    return {
        "time": q_time,
        "x": np.array([pd.Timestamp(t).month for t in q_time], dtype=int),
        "year": np.array([pd.Timestamp(t).year for t in q_time], dtype=int),
        "p05": np.asarray(q["p05"], dtype=float).ravel(),
        "p25": np.asarray(q["p25"], dtype=float).ravel(),
        "p50": np.asarray(q["median"], dtype=float).ravel(),
        "p75": np.asarray(q["p75"], dtype=float).ravel(),
        "p95": np.asarray(q["p95"], dtype=float).ravel(),
    }


def rebuild_region_anom_series_yearly():
    """
    Rebuild clean monthly anomaly series for all regions.
    No El Niño filtering.
    """

    out = {}

    for region_name in regions:

        ds_reg = era5_anom_data[region_name]

        if "valid_time" in ds_reg.dims or "valid_time" in ds_reg.coords:
            ds_reg = ds_reg.rename({"valid_time": "time"})

        t2m_series = clean_monthly_series(
            area_weighted_mean(ds_reg["t2m"])
        )

        tp_series = clean_monthly_series(
            area_weighted_mean(ds_reg["tp"])
        )

        t2m_series = standardize_monthly_time(t2m_series)
        tp_series = standardize_monthly_time(tp_series)

        t2m_series = t2m_series.sel(
            time=slice(None, observed_end)
        )

        tp_series = tp_series.sel(
            time=slice(None, observed_end)
        )

        out[region_name] = {
            "t2m": t2m_series,
            "tp": tp_series
        }

    return out


def build_forecast_quantiles_by_region_yearly():
    """
    Build forecast quantiles by region for all available ECMWF forecast months.
    """

    out = {}

    for region_name, (latN, lonW, latS, lonE) in regions.items():

        ens_t2_fc, ens_tp_fc, tp_units = get_forecast_series_for_region(
            region_name,
            latN,
            lonW,
            latS,
            lonE
        )

        ens_tp_fc = convert_forecast_precip_for_plot(
            ens_tp_fc,
            tp_units
        )

        q_t = quantile_output_to_arrays(
            ensemble_quantiles(ens_t2_fc)
        )

        q_p = quantile_output_to_arrays(
            ensemble_quantiles(ens_tp_fc)
        )

        out[region_name] = {
            "t2m": q_t,
            "tp": q_p
        }

    return out


def get_available_observed_years(region_anom_series_yearly):
    """
    Determine observed years available in the anomaly data.
    """

    all_years = []

    for region_name in regions:

        time_vals = pd.DatetimeIndex(
            pd.to_datetime(
                region_anom_series_yearly[region_name]["t2m"].time.values
            )
        )

        all_years.extend([t.year for t in time_vals])

    return sorted(list(set(all_years)))


def build_year_row(
    region_name,
    variable,
    year,
    region_anom_series_yearly,
    forecast_quantiles_by_region_yearly
):
    """
    Build one Jan-Dec row.

    Observed ERA5 fills available months.
    Forecast median fills forecast months.
    If both exist, observed is kept.
    """

    row = np.full(12, np.nan)

    year_months = calendar_year_months(year)

    obs_months = pd.DatetimeIndex([
        m for m in year_months
        if m <= observed_end
    ])

    if len(obs_months) > 0:

        observed_vals = (
            region_anom_series_yearly[region_name][variable]
            .reindex(time=obs_months)
            .values
            .astype(float)
        )

        for m, val in zip(obs_months, observed_vals):
            row[m.month - 1] = val

    q = forecast_quantiles_by_region_yearly[region_name][variable]

    if q is not None and len(q["time"]) > 0:

        for t, x, val in zip(
            q["time"],
            q["x"],
            q["p50"]
        ):

            t = pd.Timestamp(t).to_period("M").to_timestamp()

            if t.year == year:

                if t > observed_end or not np.isfinite(row[x - 1]):
                    row[x - 1] = val

    return row


def build_yearly_matrices(
    region_name,
    region_anom_series_yearly,
    forecast_quantiles_by_region_yearly,
    all_plot_years
):
    """
    Build year-on-year matrices for temperature and precipitation.
    """

    labels = []
    t_rows = []
    p_rows = []

    for year in all_plot_years:

        t_row = build_year_row(
            region_name,
            "t2m",
            year,
            region_anom_series_yearly,
            forecast_quantiles_by_region_yearly
        )

        p_row = build_year_row(
            region_name,
            "tp",
            year,
            region_anom_series_yearly,
            forecast_quantiles_by_region_yearly
        )

        if np.isfinite(t_row).any() or np.isfinite(p_row).any():

            labels.append(str(year))
            t_rows.append(t_row)
            p_rows.append(p_row)

    if len(t_rows) == 0:

        t_matrix = np.empty((0, 12), dtype=float)
        p_matrix = np.empty((0, 12), dtype=float)

    else:

        t_matrix = np.vstack(t_rows)
        p_matrix = np.vstack(p_rows)

    return labels, t_matrix, p_matrix


def rebuild_historical_el_nino_events():
    """
    Rebuild formal historical El Niño events.
    Used only for Figure 3.

    If no current/recent event was detected, all detected events are historical.
    If a current/recent event exists, only events ending before it are historical.
    """

    oni_series_tmp = nino34_running.to_series().dropna()

    oni_series_tmp.index = (
        pd.DatetimeIndex(pd.to_datetime(oni_series_tmp.index))
        .to_period("M")
        .to_timestamp()
    )

    events = detect_el_nino_events(
        oni_series_tmp,
        threshold=threshold,
        min_duration=min_duration
    )

    if ongoing_event_start is None:
        return events

    ongoing_start = pd.Timestamp(ongoing_event_start).to_period("M").to_timestamp()

    hist_events = [
        (start, end)
        for start, end in events
        if pd.Timestamp(end).to_period("M").to_timestamp() < ongoing_start
    ]

    return hist_events


def build_el_nino_event_year_records(historical_events):
    """
    Build event-year rows for historical El Niño years only.
    Used only for Figure 3.
    """

    records = []

    for start, end in historical_events:

        start = pd.Timestamp(start).to_period("M").to_timestamp()
        end = pd.Timestamp(end).to_period("M").to_timestamp()

        event_label = event_label_from_dates(start, end)

        for year in range(start.year, end.year + 1):

            year_months = calendar_year_months(year)

            event_mask = np.array([
                (m >= start) and (m <= end)
                for m in year_months
            ])

            if event_mask.any():

                records.append({
                    "event_label": event_label,
                    "year": year,
                    "months": year_months,
                    "event_mask": event_mask,
                    "start": start,
                    "end": end
                })

    return records


def build_region_el_nino_event_year_anoms(
    region_anom_series_yearly,
    event_year_records
):
    """
    Build Jan-Dec matrices for historical El Niño event-years only.
    Used only for Figure 3.
    """

    out = {
        region_name: []
        for region_name in regions
    }

    for record in event_year_records:

        year_months = record["months"]
        event_mask = record["event_mask"]

        for region_name, series_dict in region_anom_series_yearly.items():

            t_vals = (
                series_dict["t2m"]
                .reindex(time=year_months)
                .values
                .astype(float)
            )

            p_vals = (
                series_dict["tp"]
                .reindex(time=year_months)
                .values
                .astype(float)
            )

            t_vals[~event_mask] = np.nan
            p_vals[~event_mask] = np.nan

            out[region_name].append({
                "event_label": record["event_label"],
                "year": record["year"],
                "months": year_months,
                "event_mask": event_mask,
                "t2m": t_vals,
                "tp": p_vals,
                "start": record["start"],
                "end": record["end"]
            })

    return out


def latest_3_forecast_months():
    """
    Return the final 3 available forecast months from fc_times.
    Keeps full timestamps, not only month numbers.
    """

    if fc_times is None or len(fc_times) == 0:
        return pd.DatetimeIndex([])

    fc_months = (
        pd.DatetimeIndex(pd.to_datetime(fc_times))
        .to_period("M")
        .to_timestamp()
    )

    fc_months = pd.DatetimeIndex(
        sorted(fc_months.unique())
    )

    return fc_months[-3:]


# ------------------------------------------------------------
# 13C. Rebuild clean inputs
# ------------------------------------------------------------

region_anom_series_yearly = rebuild_region_anom_series_yearly()

forecast_quantiles_by_region_yearly = build_forecast_quantiles_by_region_yearly()

observed_years = get_available_observed_years(
    region_anom_series_yearly
)

min_plot_year = min(observed_years)

all_plot_years = list(
    range(
        min_plot_year,
        max_plot_year + 1
    )
)

historical_el_nino_events = rebuild_historical_el_nino_events()

el_nino_event_year_records = build_el_nino_event_year_records(
    historical_el_nino_events
)

region_el_nino_event_year_anoms = build_region_el_nino_event_year_anoms(
    region_anom_series_yearly,
    el_nino_event_year_records
)

print("\nYear-on-year heatmap rows cover:")
print(f"  {min_plot_year} to {max_plot_year}")

print("\nFigure 3 stays El Niño-specific.")
print(f"Historical El Niño events used: {len(historical_el_nino_events)}")

print("\nDiagnostics for precipitation input:")

for region_name in regions:

    p_vals_check = np.asarray(
        region_anom_series_yearly[region_name]["tp"].values,
        dtype=float
    )

    print(
        f"  {region_name}: "
        f"finite precip cells = {np.isfinite(p_vals_check).sum()}, "
        f"range = "
        f"{np.nanmin(p_vals_check) if np.isfinite(p_vals_check).any() else np.nan:.2f} to "
        f"{np.nanmax(p_vals_check) if np.isfinite(p_vals_check).any() else np.nan:.2f}"
    )


# ============================================================
# 14. FIGURE 2
# Year-on-year heatmaps, no El Niño filtering
# ============================================================

for region_name in regions:

    labels_plot, t_matrix_plot, p_matrix_plot = build_yearly_matrices(
        region_name,
        region_anom_series_yearly,
        forecast_quantiles_by_region_yearly,
        all_plot_years
    )

    if len(labels_plot) == 0:

        print(f"No data available for {region_name}. Skipping heatmap.")
        continue

    t_abs = np.nanmax(np.abs(t_matrix_plot))
    p_abs = np.nanmax(np.abs(p_matrix_plot))

    if not np.isfinite(t_abs) or t_abs == 0:
        t_abs = 1

    if not np.isfinite(p_abs) or p_abs == 0:
        p_abs = 1

    t_cmap = plt.get_cmap("RdBu_r").copy()
    p_cmap = plt.get_cmap("BrBG").copy()

    t_cmap.set_bad("white")
    p_cmap.set_bad("white")

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(14, max(6, 0.30 * len(labels_plot))),
        sharey=True
    )

    ax_t, ax_p = axes

    im_t = ax_t.imshow(
        np.ma.masked_invalid(t_matrix_plot),
        aspect="auto",
        cmap=t_cmap,
        vmin=-t_abs,
        vmax=t_abs
    )

    im_p = ax_p.imshow(
        np.ma.masked_invalid(p_matrix_plot),
        aspect="auto",
        cmap=p_cmap,
        vmin=-p_abs,
        vmax=p_abs
    )

    ax_t.set_title(f"{region_name}: temperature anomaly")
    ax_p.set_title(f"{region_name}: precipitation anomaly")

    ax_t.set_xticks(np.arange(12))
    ax_p.set_xticks(np.arange(12))

    ax_t.set_xticklabels(month_labels)
    ax_p.set_xticklabels(month_labels)

    ax_t.set_yticks(np.arange(len(labels_plot)))
    ax_t.set_yticklabels(labels_plot)

    ax_p.set_yticks(np.arange(len(labels_plot)))
    ax_p.set_yticklabels(labels_plot)

    for ax in axes:
        ax.set_xlabel("Calendar month")

    # Highlight forecast year rows, e.g. 2026
    if len(forecast_years_available) > 0:

        for forecast_year in forecast_years_available:

            if str(forecast_year) in labels_plot:

                row_idx = labels_plot.index(str(forecast_year))

                for ax in axes:

                    ax.axhline(
                        row_idx - 0.5,
                        color="black",
                        linewidth=1.3
                    )

                    ax.axhline(
                        row_idx + 0.5,
                        color="black",
                        linewidth=1.3
                    )

    cbar_t = fig.colorbar(
        im_t,
        ax=ax_t,
        fraction=0.046,
        pad=0.04
    )

    cbar_p = fig.colorbar(
        im_p,
        ax=ax_p,
        fraction=0.046,
        pad=0.04
    )

    cbar_t.set_label("°C anomaly")
    cbar_p.set_label("mm/month anomaly")

    if len(forecast_years_available) > 0:
        forecast_label = ", ".join([str(y) for y in forecast_years_available])
        subtitle = f"forecast row(s): {forecast_label}"
    else:
        subtitle = "observed years only"

    fig.suptitle(
        f"{region_name}: year-on-year anomalies "
        f"(ERA5 observed + {FORECAST_LABEL} forecast; {subtitle})",
        y=1.02,
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 15. FIGURE 3
# Final 3 forecast months vs historical El Niño years only
# ============================================================

summary_fc_months = latest_3_forecast_months()

if len(summary_fc_months) == 0:

    print("\nNo forecast months available for Figure 3.")

else:

    summary_months = [
        pd.Timestamp(t).month
        for t in summary_fc_months
    ]

    summary_month_labels = [
        pd.Timestamp(t).strftime("%b %Y")
        for t in summary_fc_months
    ]

    last3_label = " - ".join(summary_month_labels)

    print("\nFigure 3 final forecast months used:", summary_month_labels)
    print("Figure 3 is El Niño-specific.")
    print("It compares the final 3 forecast months against the same calendar months in historical El Niño event-years only.")

    month_idx = np.array(summary_months) - 1

    summary_rows = []

    for region_name in regions:

        records = region_el_nino_event_year_anoms[region_name]

        hist_t_last3 = []
        hist_p_last3 = []

        for record in records:

            t_vals = np.asarray(
                record["t2m"],
                dtype=float
            )

            p_vals = np.asarray(
                record["tp"],
                dtype=float
            )

            t_subset = t_vals[month_idx]
            p_subset = p_vals[month_idx]

            if np.isfinite(t_subset).any():
                hist_t_last3.append(
                    np.nanmean(t_subset)
                )

            if np.isfinite(p_subset).any():
                hist_p_last3.append(
                    np.nanmean(p_subset)
                )

        hist_t_last3 = np.array(
            hist_t_last3,
            dtype=float
        )

        hist_p_last3 = np.array(
            hist_p_last3,
            dtype=float
        )

        t_fc = forecast_quantiles_by_region_yearly[region_name]["t2m"]
        p_fc = forecast_quantiles_by_region_yearly[region_name]["tp"]

        t_latest_vals = []
        p_latest_vals = []

        for t, val in zip(
            t_fc["time"],
            t_fc["p50"]
        ):

            t = pd.Timestamp(t).to_period("M").to_timestamp()

            if t in summary_fc_months:
                t_latest_vals.append(val)

        for t, val in zip(
            p_fc["time"],
            p_fc["p50"]
        ):

            t = pd.Timestamp(t).to_period("M").to_timestamp()

            if t in summary_fc_months:
                p_latest_vals.append(val)

        t_latest_vals = np.array(
            t_latest_vals,
            dtype=float
        )

        p_latest_vals = np.array(
            p_latest_vals,
            dtype=float
        )

        if np.isfinite(t_latest_vals).any():
            t_latest = np.nanmean(t_latest_vals)
        else:
            t_latest = np.nan

        if np.isfinite(p_latest_vals).any():
            p_latest = np.nanmean(p_latest_vals)
        else:
            p_latest = np.nan

        if len(hist_t_last3) > 0 and np.isfinite(t_latest):
            t_percentile = 100 * np.mean(
                hist_t_last3 <= t_latest
            )
        else:
            t_percentile = np.nan

        if len(hist_p_last3) > 0 and np.isfinite(p_latest):
            p_percentile = 100 * np.mean(
                hist_p_last3 <= p_latest
            )
        else:
            p_percentile = np.nan

        if len(hist_t_last3) > 0:
            t_hist_mean = np.nanmean(hist_t_last3)
            t_hist_median = np.nanmedian(hist_t_last3)
            t_hist_p25 = np.nanpercentile(hist_t_last3, 25)
            t_hist_p75 = np.nanpercentile(hist_t_last3, 75)
        else:
            t_hist_mean = np.nan
            t_hist_median = np.nan
            t_hist_p25 = np.nan
            t_hist_p75 = np.nan

        if len(hist_p_last3) > 0:
            p_hist_mean = np.nanmean(hist_p_last3)
            p_hist_median = np.nanmedian(hist_p_last3)
            p_hist_p25 = np.nanpercentile(hist_p_last3, 25)
            p_hist_p75 = np.nanpercentile(hist_p_last3, 75)
        else:
            p_hist_mean = np.nan
            p_hist_median = np.nan
            p_hist_p25 = np.nan
            p_hist_p75 = np.nan

        summary_rows.append({
            "region": region_name,
            "final_3_forecast_months": last3_label,

            "t_latest_forecast_last3": t_latest,
            "t_hist_mean_el_nino": t_hist_mean,
            "t_hist_median_el_nino": t_hist_median,
            "t_hist_p25_el_nino": t_hist_p25,
            "t_hist_p75_el_nino": t_hist_p75,
            "t_percentile_vs_el_nino": t_percentile,

            "p_latest_forecast_last3": p_latest,
            "p_hist_mean_el_nino": p_hist_mean,
            "p_hist_median_el_nino": p_hist_median,
            "p_hist_p25_el_nino": p_hist_p25,
            "p_hist_p75_el_nino": p_hist_p75,
            "p_percentile_vs_el_nino": p_percentile,

            "n_hist_t_el_nino_years": len(hist_t_last3),
            "n_hist_p_el_nino_years": len(hist_p_last3)
        })

    summary_df = pd.DataFrame(summary_rows)

    print("\nFinal 3 forecast months El Niño comparison summary:")
    display(summary_df)

    # --------------------------------------------------------
    # 15B. Plot final 3 forecast months comparison bars
    # --------------------------------------------------------

    regions_order = summary_df["region"].values
    x = np.arange(len(regions_order))

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(13, 4),
        sharex=True
    )

    ax_t, ax_p = axes

    # Clean numeric arrays
    t_hist_med = summary_df["t_hist_median_el_nino"].astype(float).values
    t_hist_p25 = summary_df["t_hist_p25_el_nino"].astype(float).values
    t_hist_p75 = summary_df["t_hist_p75_el_nino"].astype(float).values
    t_latest = summary_df["t_latest_forecast_last3"].astype(float).values

    p_hist_med = summary_df["p_hist_median_el_nino"].astype(float).values
    p_hist_p25 = summary_df["p_hist_p25_el_nino"].astype(float).values
    p_hist_p75 = summary_df["p_hist_p75_el_nino"].astype(float).values
    p_latest = summary_df["p_latest_forecast_last3"].astype(float).values

    # Positive asymmetric IQR distances around median
    t_yerr_low = np.maximum(t_hist_med - t_hist_p25, 0)
    t_yerr_high = np.maximum(t_hist_p75 - t_hist_med, 0)

    p_yerr_low = np.maximum(p_hist_med - p_hist_p25, 0)
    p_yerr_high = np.maximum(p_hist_p75 - p_hist_med, 0)

    # Replace non-finite yerr values with 0 so matplotlib does not crash
    t_yerr_low = np.where(np.isfinite(t_yerr_low), t_yerr_low, 0)
    t_yerr_high = np.where(np.isfinite(t_yerr_high), t_yerr_high, 0)

    p_yerr_low = np.where(np.isfinite(p_yerr_low), p_yerr_low, 0)
    p_yerr_high = np.where(np.isfinite(p_yerr_high), p_yerr_high, 0)

    # Temperature bars
    ax_t.bar(
        x - 0.18,
        t_hist_med,
        width=0.35,
        color="lightcoral",
        label="Historical El Niño median"
    )

    ax_t.bar(
        x + 0.18,
        t_latest,
        width=0.35,
        color="black",
        label=f"{FORECAST_SHORT_LABEL} final 3 forecast months"
    )

    ax_t.errorbar(
        x - 0.18,
        t_hist_med,
        yerr=np.vstack([t_yerr_low, t_yerr_high]),
        fmt="none",
        ecolor="red",
        capsize=4,
        alpha=0.7
    )

    ax_t.axhline(
        0,
        color="gray",
        linewidth=0.8
    )

    ax_t.set_title(
        f"{last3_label} temperature anomaly"
    )

    ax_t.set_ylabel("°C anomaly")

    ax_t.set_xticks(x)

    ax_t.set_xticklabels(
        regions_order,
        rotation=30,
        ha="right"
    )

    ax_t.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.35
    )

    # Precipitation bars
    ax_p.bar(
        x - 0.18,
        p_hist_med,
        width=0.35,
        color="lightskyblue",
        label="Historical El Niño median"
    )

    ax_p.bar(
        x + 0.18,
        p_latest,
        width=0.35,
        color="black",
        label=f"{FORECAST_SHORT_LABEL} final 3 forecast months"
    )

    ax_p.errorbar(
        x - 0.18,
        p_hist_med,
        yerr=np.vstack([p_yerr_low, p_yerr_high]),
        fmt="none",
        ecolor="blue",
        capsize=4,
        alpha=0.7
    )

    ax_p.axhline(
        0,
        color="gray",
        linewidth=0.8
    )

    ax_p.set_title(
        f"{last3_label} precipitation anomaly"
    )

    ax_p.set_ylabel("mm/month anomaly")

    ax_p.set_xticks(x)

    ax_p.set_xticklabels(
        regions_order,
        rotation=30,
        ha="right"
    )

    ax_p.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.35
    )

    handles, labels = ax_t.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.10)
    )

    fig.suptitle(
        f"Final 3 forecast months compared with historical El Niño years: {last3_label}",
        y=1.18,
        fontsize=13
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 16. PROVIDER COMPARISON
# ECMWF SEAS5 vs NOAA/NCEP CFSv2
#
# Purpose:
#   - Keep ECMWF as the main provider for all figures above.
#   - Load NOAA only here.
#   - Compare ECMWF and NOAA only over the final 3 common forecast months.
#   - Do not combine providers in the same forecast plot or uncertainty envelope.
#   - Do not plot the final ECMWF-minus-NOAA difference chart.
# ============================================================

# ------------------------------------------------------------
# 16A. Safety check: make sure ECMWF tables exist
# ------------------------------------------------------------

activate_loaded_forecast(ecmwf_result)

if "ecmwf_forecast_tables" not in globals():
    print("ECMWF forecast tables not found. Rebuilding them from active ECMWF forecast.")
    ecmwf_forecast_tables, ecmwf_combined_forecast_table = build_forecast_tables_for_active_source()

if "forecast_tables" not in globals():
    forecast_tables = ecmwf_forecast_tables

print("\nECMWF forecast months available by country:")

for region_name, df in ecmwf_forecast_tables.items():

    df_tmp = df.copy()

    if "Month_dt" not in df_tmp.columns:
        df_tmp["Month_dt"] = (
            pd.DatetimeIndex(
                pd.to_datetime(
                    df_tmp["Month"],
                    format="%B %Y"
                )
            )
            .to_period("M")
            .to_timestamp()
        )

    else:
        df_tmp["Month_dt"] = (
            pd.DatetimeIndex(pd.to_datetime(df_tmp["Month_dt"]))
            .to_period("M")
            .to_timestamp()
        )

    print(
        region_name,
        df_tmp["Month_dt"].dt.strftime("%Y-%m").tolist()
    )


# ------------------------------------------------------------
# 16B. Load NOAA separately
# ------------------------------------------------------------

print("\nLoading NOAA/NCEP CFSv2 for final provider comparison only.")

set_forecast_source(COMPARISON_FORECAST_SOURCE)

noaa_result = load_latest_forecast_for_active_source()
activate_loaded_forecast(noaa_result)

noaa_forecast_tables, noaa_combined_forecast_table = build_forecast_tables_for_active_source()

print("\nNOAA forecast months available by country:")

for region_name, df in noaa_forecast_tables.items():

    df_tmp = df.copy()

    if "Month_dt" not in df_tmp.columns:
        df_tmp["Month_dt"] = (
            pd.DatetimeIndex(
                pd.to_datetime(
                    df_tmp["Month"],
                    format="%B %Y"
                )
            )
            .to_period("M")
            .to_timestamp()
        )

    else:
        df_tmp["Month_dt"] = (
            pd.DatetimeIndex(pd.to_datetime(df_tmp["Month_dt"]))
            .to_period("M")
            .to_timestamp()
        )

    print(
        region_name,
        df_tmp["Month_dt"].dt.strftime("%Y-%m").tolist()
    )


# ------------------------------------------------------------
# 16C. Helper functions for provider comparison
# ------------------------------------------------------------

def ensure_month_dt(df):
    """
    Ensure a dataframe has a clean Month_dt column as monthly-start timestamps.
    """

    df = df.copy()

    if "Month_dt" in df.columns:

        df["Month_dt"] = (
            pd.DatetimeIndex(pd.to_datetime(df["Month_dt"]))
            .to_period("M")
            .to_timestamp()
        )

    else:

        df["Month_dt"] = (
            pd.DatetimeIndex(
                pd.to_datetime(
                    df["Month"],
                    format="%B %Y"
                )
            )
            .to_period("M")
            .to_timestamp()
        )

    return df


def get_common_months_for_region(ecmwf_tables, noaa_tables, region_name):
    """
    Find forecast months available in both providers for one region.
    """

    ecmwf_df = ensure_month_dt(ecmwf_tables[region_name])
    noaa_df = ensure_month_dt(noaa_tables[region_name])

    ecmwf_months = pd.DatetimeIndex(
        sorted(ecmwf_df["Month_dt"].unique())
    )

    noaa_months = pd.DatetimeIndex(
        sorted(noaa_df["Month_dt"].unique())
    )

    common_months = pd.DatetimeIndex(
        sorted(ecmwf_months.intersection(noaa_months))
    )

    return common_months


def get_final_n_common_months(ecmwf_tables, noaa_tables, region_name, n_months=3):
    """
    Return the final n common forecast months between ECMWF and NOAA for one region.
    This ensures both providers are compared over the same calendar months.
    """

    common_months = get_common_months_for_region(
        ecmwf_tables,
        noaa_tables,
        region_name
    )

    if len(common_months) == 0:
        return pd.DatetimeIndex([])

    return common_months[-n_months:]


def summarize_provider_region_for_months(
    provider_name,
    region_name,
    forecast_tables_in,
    months_to_use
):
    """
    Summarize one provider and one region over a set of selected forecast months.
    """

    df = ensure_month_dt(forecast_tables_in[region_name])

    months_to_use = (
        pd.DatetimeIndex(pd.to_datetime(months_to_use))
        .to_period("M")
        .to_timestamp()
    )

    df_sel = (
        df
        .loc[df["Month_dt"].isin(months_to_use)]
        .sort_values("Month_dt")
        .copy()
    )

    if len(df_sel) == 0:

        return {
            "Provider": provider_name,
            "Country": region_name,
            "Months compared": "",
            "Number of months": 0,
            "Mean temperature anomaly (°C)": np.nan,
            "Mean precipitation anomaly (mm/month)": np.nan,
            "Min temperature anomaly (°C)": np.nan,
            "Max temperature anomaly (°C)": np.nan,
            "Min precipitation anomaly (mm/month)": np.nan,
            "Max precipitation anomaly (mm/month)": np.nan,
        }

    return {
        "Provider": provider_name,
        "Country": region_name,
        "Months compared": ", ".join(df_sel["Month_dt"].dt.strftime("%b %Y").tolist()),
        "Number of months": len(df_sel),
        "Mean temperature anomaly (°C)": df_sel["Temperature anomaly (°C)"].mean(),
        "Mean precipitation anomaly (mm/month)": df_sel["Precipitation anomaly (mm/month)"].mean(),
        "Min temperature anomaly (°C)": df_sel["Temperature anomaly (°C)"].min(),
        "Max temperature anomaly (°C)": df_sel["Temperature anomaly (°C)"].max(),
        "Min precipitation anomaly (mm/month)": df_sel["Precipitation anomaly (mm/month)"].min(),
        "Max precipitation anomaly (mm/month)": df_sel["Precipitation anomaly (mm/month)"].max(),
    }


# ------------------------------------------------------------
# 16D. Build provider-comparison summary
# Final 3 common forecast months only
# ------------------------------------------------------------

FINAL_PROVIDER_COMPARISON_MONTHS = 3

comparison_rows = []

for region_name in regions:

    final_common_months = get_final_n_common_months(
        ecmwf_forecast_tables,
        noaa_forecast_tables,
        region_name,
        n_months=FINAL_PROVIDER_COMPARISON_MONTHS
    )

    print(
        f"\n{region_name} final {FINAL_PROVIDER_COMPARISON_MONTHS} common ECMWF/NOAA months:",
        final_common_months.strftime("%Y-%m").tolist()
    )

    comparison_rows.append(
        summarize_provider_region_for_months(
            provider_name="ECMWF",
            region_name=region_name,
            forecast_tables_in=ecmwf_forecast_tables,
            months_to_use=final_common_months
        )
    )

    comparison_rows.append(
        summarize_provider_region_for_months(
            provider_name="NOAA",
            region_name=region_name,
            forecast_tables_in=noaa_forecast_tables,
            months_to_use=final_common_months
        )
    )

provider_comparison_summary = pd.DataFrame(comparison_rows)

numeric_cols = [
    "Mean temperature anomaly (°C)",
    "Mean precipitation anomaly (mm/month)",
    "Min temperature anomaly (°C)",
    "Max temperature anomaly (°C)",
    "Min precipitation anomaly (mm/month)",
    "Max precipitation anomaly (mm/month)"
]

provider_comparison_summary[numeric_cols] = provider_comparison_summary[numeric_cols].round(2)

print("\nECMWF vs NOAA provider-comparison summary:")
print(f"Comparison uses final {FINAL_PROVIDER_COMPARISON_MONTHS} common forecast months only.")
display(provider_comparison_summary)


# ------------------------------------------------------------
# 16E. Difference table: ECMWF minus NOAA
# Table only, no difference plot
# ------------------------------------------------------------

difference_rows = []

for region_name in regions:

    comp_region = provider_comparison_summary[
        provider_comparison_summary["Country"] == region_name
    ].copy()

    ecmwf_row = comp_region[comp_region["Provider"] == "ECMWF"]
    noaa_row = comp_region[comp_region["Provider"] == "NOAA"]

    if len(ecmwf_row) == 0 or len(noaa_row) == 0:
        continue

    ecmwf_row = ecmwf_row.iloc[0]
    noaa_row = noaa_row.iloc[0]

    difference_rows.append({
        "Country": region_name,
        "Months compared": ecmwf_row["Months compared"],
        "Temperature difference ECMWF - NOAA (°C)": (
            ecmwf_row["Mean temperature anomaly (°C)"]
            - noaa_row["Mean temperature anomaly (°C)"]
        ),
        "Precipitation difference ECMWF - NOAA (mm/month)": (
            ecmwf_row["Mean precipitation anomaly (mm/month)"]
            - noaa_row["Mean precipitation anomaly (mm/month)"]
        ),
    })

provider_difference_summary = pd.DataFrame(difference_rows)

if len(provider_difference_summary) > 0:

    provider_difference_summary[
        [
            "Temperature difference ECMWF - NOAA (°C)",
            "Precipitation difference ECMWF - NOAA (mm/month)"
        ]
    ] = provider_difference_summary[
        [
            "Temperature difference ECMWF - NOAA (°C)",
            "Precipitation difference ECMWF - NOAA (mm/month)"
        ]
    ].round(2)

print("\nProvider difference summary:")
print("Positive values mean ECMWF is higher than NOAA.")
display(provider_difference_summary)


# ------------------------------------------------------------
# 16F. Bar chart comparing provider means
# Keep this chart
# ------------------------------------------------------------

if len(provider_comparison_summary) > 0:

    regions_order = list(regions.keys())
    x = np.arange(len(regions_order))
    width = 0.35

    ecmwf_plot = (
        provider_comparison_summary
        .query("Provider == 'ECMWF'")
        .set_index("Country")
        .reindex(regions_order)
    )

    noaa_plot = (
        provider_comparison_summary
        .query("Provider == 'NOAA'")
        .set_index("Country")
        .reindex(regions_order)
    )

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(13, 4.5),
        sharex=True
    )

    ax_t, ax_p = axes

    # Temperature comparison
    ax_t.bar(
        x - width / 2,
        ecmwf_plot["Mean temperature anomaly (°C)"].values,
        width=width,
        color="firebrick",
        alpha=0.75,
        label="ECMWF"
    )

    ax_t.bar(
        x + width / 2,
        noaa_plot["Mean temperature anomaly (°C)"].values,
        width=width,
        color="black",
        alpha=0.75,
        label="NOAA"
    )

    ax_t.axhline(
        0,
        color="gray",
        linewidth=0.8
    )

    ax_t.set_title("Mean temperature anomaly")
    ax_t.set_ylabel("°C")
    ax_t.set_xticks(x)

    ax_t.set_xticklabels(
        regions_order,
        rotation=30,
        ha="right"
    )

    ax_t.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.35
    )

    # Precipitation comparison
    ax_p.bar(
        x - width / 2,
        ecmwf_plot["Mean precipitation anomaly (mm/month)"].values,
        width=width,
        color="royalblue",
        alpha=0.75,
        label="ECMWF"
    )

    ax_p.bar(
        x + width / 2,
        noaa_plot["Mean precipitation anomaly (mm/month)"].values,
        width=width,
        color="black",
        alpha=0.75,
        label="NOAA"
    )

    ax_p.axhline(
        0,
        color="gray",
        linewidth=0.8
    )

    ax_p.set_title("Mean precipitation anomaly")
    ax_p.set_ylabel("mm/month")
    ax_p.set_xticks(x)

    ax_p.set_xticklabels(
        regions_order,
        rotation=30,
        ha="right"
    )

    ax_p.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.35
    )

    handles, labels = ax_t.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.07)
    )

    all_month_labels = sorted(
        set(
            provider_comparison_summary["Months compared"]
            .dropna()
            .astype(str)
            .tolist()
        )
    )

    if len(all_month_labels) == 1:
        month_text = all_month_labels[0]
    else:
        month_text = "region-specific final 3 common months"

    fig.suptitle(
        f"ECMWF vs NOAA forecast anomaly comparison\n"
        f"Final {FINAL_PROVIDER_COMPARISON_MONTHS} common months: {month_text}",
        y=1.18,
        fontsize=13
    )

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 16G. Final safety: reactivate ECMWF after NOAA comparison
# ------------------------------------------------------------

activate_loaded_forecast(ecmwf_result)

print("\nFinal active source after provider comparison:", FORECAST_LABEL)
print("Forecast months active again:", fc_times.strftime("%Y-%m").tolist())
# ============================================================
# 17. PERU SUBREGIONAL ECMWF FORECAST ANALYSIS WITH POLYGON MASKS
#
# Purpose:
#   - ECMWF only
#   - Final 3 forecast months only
#   - More accurate than rectangular boxes:
#       1. Uses polygon masks
#       2. Clips polygons to Peru boundary where possible
#       3. Area-weights only grid cells inside each polygon
#
# Notes:
#   - The subregion polygons below are approximate physiographic polygons.
#   - They are better than rectangles because they avoid obvious ocean/neighbouring-country cells.
#   - For publication-grade analysis, replace these hand-drawn polygons with official
#     ecoregion, elevation-zone, department, or watershed shapefiles.
# ============================================================

# Safety: ensure ECMWF is active, not NOAA
activate_loaded_forecast(ecmwf_result)


# ------------------------------------------------------------
# 17A. Imports and polygon utilities
# ------------------------------------------------------------

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from shapely.prepared import prep

try:
    import cartopy.io.shapereader as shpreader
    NATURAL_EARTH_AVAILABLE = True
except Exception:
    NATURAL_EARTH_AVAILABLE = False
    print("Cartopy shapereader not available. Peru boundary clipping will be skipped.")


def get_peru_boundary_geometry():
    """
    Get Peru country boundary from Natural Earth via Cartopy.
    Returns a shapely geometry in lon/lat coordinates.

    If Natural Earth is unavailable, returns None and the code will use
    the subregion polygons without country clipping.
    """

    if not NATURAL_EARTH_AVAILABLE:
        return None

    try:
        shpfilename = shpreader.natural_earth(
            resolution="50m",
            category="cultural",
            name="admin_0_countries"
        )

        reader = shpreader.Reader(shpfilename)

        peru_geoms = []

        for record in reader.records():

            attrs = record.attributes

            name_candidates = [
                attrs.get("ADMIN", ""),
                attrs.get("NAME_LONG", ""),
                attrs.get("NAME", ""),
                attrs.get("SOVEREIGNT", "")
            ]

            if any(str(name).lower() == "peru" for name in name_candidates):
                peru_geoms.append(record.geometry)

        if len(peru_geoms) == 0:
            print("Peru boundary not found in Natural Earth. Proceeding without clipping.")
            return None

        peru_geom = unary_union(peru_geoms)

        print("Peru boundary loaded from Natural Earth.")

        return peru_geom

    except Exception as e:
        print("Could not load Peru boundary from Natural Earth:", repr(e))
        print("Proceeding without country-boundary clipping.")
        return None


peru_country_geom = get_peru_boundary_geometry()


# ------------------------------------------------------------
# 17B. Define approximate Peru subregion polygons
#
# Coordinates are lon/lat.
# These polygons are rough estimates
# They are clipped to Peru boundary if Natural Earth boundary is available.
# ------------------------------------------------------------

peru_subregion_polygons_raw = {
    "North coast": Polygon([
        (-82.6, 0.2),
        (-78.8, 0.2),
        (-79.0, -3.5),
        (-80.0, -7.0),
        (-81.7, -7.0),
        (-82.6, -3.5),
        (-82.6, 0.2),
    ]),

    "Central coast": Polygon([
        (-81.7, -7.0),
        (-75.4, -7.0),
        (-76.2, -11.0),
        (-78.3, -14.5),
        (-80.5, -14.5),
        (-81.7, -10.0),
        (-81.7, -7.0),
    ]),

    "South coast": Polygon([
        (-80.5, -14.0),
        (-72.0, -14.0),
        (-69.2, -18.8),
        (-70.2, -20.5),
        (-74.5, -20.5),
        (-78.5, -17.0),
        (-80.5, -14.0),
    ]),

    "Northern Andes": Polygon([
        (-79.2, 0.2),
        (-73.1, 0.2),
        (-73.8, -4.5),
        (-75.2, -8.5),
        (-79.4, -8.5),
        (-79.0, -4.0),
        (-79.2, 0.2),
    ]),

    "Central Andes": Polygon([
        (-79.4, -8.0),
        (-70.0, -8.0),
        (-70.4, -13.5),
        (-72.4, -15.2),
        (-76.8, -15.0),
        (-78.8, -11.5),
        (-79.4, -8.0),
    ]),

    "Southern Andes": Polygon([
        (-76.8, -14.5),
        (-68.0, -14.5),
        (-68.0, -20.5),
        (-72.5, -20.5),
        (-75.5, -18.0),
        (-76.8, -14.5),
    ]),

    "Amazon north": Polygon([
        (-75.0, 0.2),
        (-66.0, 0.2),
        (-66.0, -7.5),
        (-70.8, -8.5),
        (-73.8, -4.5),
        (-75.0, 0.2),
    ]),

    "Amazon south": Polygon([
        (-73.8, -7.5),
        (-66.0, -7.5),
        (-66.0, -14.8),
        (-69.5, -14.8),
        (-72.0, -13.5),
        (-74.8, -10.0),
        (-73.8, -7.5),
    ]),
}


def clip_subregions_to_peru(raw_polygons, peru_geom=None):
    """
    Clip subregion polygons to Peru boundary if possible.
    Drops empty polygons.
    """

    out = {}

    for name, geom in raw_polygons.items():

        if peru_geom is not None:
            clipped = geom.intersection(peru_geom)
        else:
            clipped = geom

        if clipped.is_empty:
            print(f"Warning: {name} polygon is empty after clipping. Skipping.")
            continue

        out[name] = clipped

    return out


peru_subregion_polygons = clip_subregions_to_peru(
    peru_subregion_polygons_raw,
    peru_country_geom
)

print("\nPeru polygon subregions used:")
for name, geom in peru_subregion_polygons.items():
    print(f"  {name}: area-like geometry bounds = {geom.bounds}")


# ------------------------------------------------------------
# 17C. Helper functions for polygon masking and weighted means
# ------------------------------------------------------------

def lon_values_for_mask(da, lon_name):
    """
    Return longitude values in -180 to 180 range for shapely masking.
    The DataArray itself keeps its original lon coordinate.
    """

    lon_vals = np.asarray(da[lon_name].values, dtype=float)

    if np.nanmax(lon_vals) > 180:
        lon_vals_for_mask = ((lon_vals + 180) % 360) - 180
    else:
        lon_vals_for_mask = lon_vals

    return lon_vals_for_mask


def polygon_mask_for_dataarray(da, polygon):
    """
    Create a 2D boolean mask for grid-cell centres inside a shapely polygon.
    Returns mask with dimensions [lat, lon].
    """

    lat_name, lon_name = get_lat_lon_names(da)

    lat_vals = np.asarray(da[lat_name].values, dtype=float)
    lon_vals_raw = np.asarray(da[lon_name].values, dtype=float)
    lon_vals_mask = lon_values_for_mask(da, lon_name)

    lon2d, lat2d = np.meshgrid(lon_vals_mask, lat_vals)

    polygon_prepared = prep(polygon)

    mask_bool = np.zeros(lon2d.shape, dtype=bool)

    for i in range(lat2d.shape[0]):
        for j in range(lat2d.shape[1]):
            point = Point(float(lon2d[i, j]), float(lat2d[i, j]))
            mask_bool[i, j] = polygon_prepared.contains(point) or polygon_prepared.touches(point)

    mask = xr.DataArray(
        mask_bool,
        coords={
            lat_name: da[lat_name],
            lon_name: da[lon_name]
        },
        dims=[lat_name, lon_name],
        name="polygon_mask"
    )

    return mask


def polygon_area_weighted_mean(da, polygon):
    """
    Area-weighted mean over grid cells inside a polygon.
    Preserves non-spatial dimensions such as time and ensemble member number.
    """

    lat_name, lon_name = get_lat_lon_names(da)

    mask = polygon_mask_for_dataarray(da, polygon)

    n_cells = int(mask.sum().values)

    if n_cells == 0:
        raise ValueError("Polygon contains no forecast grid-cell centres.")

    da_masked = da.where(mask)

    weights = np.cos(np.deg2rad(da[lat_name]))

    weighted_mean = da_masked.weighted(weights).mean(
        dim=[lat_name, lon_name],
        skipna=True
    )

    return weighted_mean, n_cells


def get_final_forecast_months_for_peru(n_months=3):
    """
    Return final n available ECMWF forecast months from fc_times.
    """

    if fc_times is None or len(fc_times) == 0:
        return pd.DatetimeIndex([])

    months = (
        pd.DatetimeIndex(pd.to_datetime(fc_times))
        .to_period("M")
        .to_timestamp()
    )

    months = pd.DatetimeIndex(
        sorted(months.unique())
    )

    return months[-n_months:]


def standardize_series_monthly_time(da):
    """
    Standardize the time coordinate to monthly-start timestamps.
    """

    da = da.copy()

    if "valid_time" in da.dims or "valid_time" in da.coords:
        da = da.rename({"valid_time": "time"})

    if "time" not in da.coords:
        raise ValueError("DataArray has no time coordinate.")

    da = da.assign_coords(
        time=(
            pd.DatetimeIndex(pd.to_datetime(da.time.values))
            .to_period("M")
            .to_timestamp()
        )
    )

    da = da.groupby("time").mean(skipna=True)

    return da


def convert_forecast_precip_to_monthly(da, tp_units):
    """
    Convert forecast precipitation anomaly to mm/month if currently mm/day.
    """

    da = da.copy()

    if tp_units == "mm/day":

        days = xr.DataArray(
            [pd.Timestamp(t).days_in_month for t in da.time.values],
            coords={"time": da.time},
            dims=["time"]
        )

        da = da * days

    return da


def one_month_ensemble_stats(da, month_date):
    """
    Extract median, p25, and p75 for one forecast month.
    If no ensemble dimension exists, the same value is returned for all stats.
    """

    da = standardize_series_monthly_time(da)

    month_date = pd.Timestamp(month_date).to_period("M").to_timestamp()

    available_months = pd.DatetimeIndex(pd.to_datetime(da.time.values))

    if month_date not in available_months:
        return np.nan, np.nan, np.nan

    da_month = da.sel(time=month_date)

    if "number" in da_month.dims:

        p25 = float(
            da_month.quantile(
                0.25,
                dim="number",
                skipna=True
            ).values
        )

        median = float(
            da_month.quantile(
                0.50,
                dim="number",
                skipna=True
            ).values
        )

        p75 = float(
            da_month.quantile(
                0.75,
                dim="number",
                skipna=True
            ).values
        )

    else:

        median = float(da_month.values)
        p25 = median
        p75 = median

    return median, p25, p75


# ------------------------------------------------------------
# 17D. Build polygon-based Peru subregional forecast table
# ------------------------------------------------------------

def build_peru_polygon_forecast_table(n_months=3):
    """
    Build a polygon-masked month-by-month ECMWF table for Peru subregions.
    Uses one Peru forecast field and applies subregion polygon masks to it.
    """

    target_months = get_final_forecast_months_for_peru(n_months=n_months)

    if len(target_months) == 0:
        raise ValueError("No forecast months available for Peru polygon analysis.")

    print(
        f"\nUsing final {n_months} {FORECAST_LABEL} forecast months for polygon Peru analysis:",
        target_months.strftime("%Y-%m").tolist()
    )

    # Get Peru forecast anomaly maps once, then mask subregions
    latN, lonW, latS, lonE = regions["Peru"]

    fc_t2_peru, fc_tp_peru, fc_tp_units = get_forecast_map_anomalies(
        "Peru",
        latN,
        lonW,
        latS,
        lonE
    )

    fc_tp_peru = convert_forecast_precip_to_monthly(
        fc_tp_peru,
        fc_tp_units
    )

    fc_t2_peru = standardize_series_monthly_time(fc_t2_peru)
    fc_tp_peru = standardize_series_monthly_time(fc_tp_peru)

    rows = []

    for subregion_name, polygon in peru_subregion_polygons.items():

        print(f"\nProcessing polygon subregion: {subregion_name}")

        try:
            t2_series, n_temp_cells = polygon_area_weighted_mean(
                fc_t2_peru,
                polygon
            )

            tp_series, n_precip_cells = polygon_area_weighted_mean(
                fc_tp_peru,
                polygon
            )

        except Exception as e:
            print(f"  Skipping {subregion_name}: {repr(e)}")
            continue

        print(
            f"  Grid cells inside polygon: temperature={n_temp_cells}, precipitation={n_precip_cells}"
        )

        for month_date in target_months:

            t_med, t_p25, t_p75 = one_month_ensemble_stats(
                t2_series,
                month_date
            )

            p_med, p_p25, p_p75 = one_month_ensemble_stats(
                tp_series,
                month_date
            )

            rows.append({
                "Subregion": subregion_name,
                "Month_dt": month_date,
                "Month": month_date.strftime("%B %Y"),
                "Temperature grid cells": n_temp_cells,
                "Precipitation grid cells": n_precip_cells,

                "Temperature median anomaly (°C)": t_med,
                "Temperature p25 anomaly (°C)": t_p25,
                "Temperature p75 anomaly (°C)": t_p75,

                "Precipitation median anomaly (mm/month)": p_med,
                "Precipitation p25 anomaly (mm/month)": p_p25,
                "Precipitation p75 anomaly (mm/month)": p_p75,
            })

    df = pd.DataFrame(rows)

    numeric_cols = [
        "Temperature median anomaly (°C)",
        "Temperature p25 anomaly (°C)",
        "Temperature p75 anomaly (°C)",
        "Precipitation median anomaly (mm/month)",
        "Precipitation p25 anomaly (mm/month)",
        "Precipitation p75 anomaly (mm/month)",
    ]

    if len(df) > 0:
        df[numeric_cols] = df[numeric_cols].round(2)

    return df


peru_polygon_forecast_table = build_peru_polygon_forecast_table(
    n_months=3
)

print("\nPolygon-based Peru subregional ECMWF forecast table:")
display(
    peru_polygon_forecast_table[
        [
            "Subregion",
            "Month",
            "Temperature grid cells",
            "Precipitation grid cells",
            "Temperature median anomaly (°C)",
            "Temperature p25 anomaly (°C)",
            "Temperature p75 anomaly (°C)",
            "Precipitation median anomaly (mm/month)",
            "Precipitation p25 anomaly (mm/month)",
            "Precipitation p75 anomaly (mm/month)",
        ]
    ]
)


# ------------------------------------------------------------
# 17E. Final-3-month polygon summary by subregion
# ------------------------------------------------------------

peru_polygon_summary = (
    peru_polygon_forecast_table
    .groupby("Subregion", as_index=False)
    .agg(
        **{
            "Temperature grid cells": (
                "Temperature grid cells",
                "max"
            ),
            "Precipitation grid cells": (
                "Precipitation grid cells",
                "max"
            ),
            "Mean temperature anomaly (°C)": (
                "Temperature median anomaly (°C)",
                "mean"
            ),
            "Max temperature anomaly (°C)": (
                "Temperature median anomaly (°C)",
                "max"
            ),
            "Min temperature anomaly (°C)": (
                "Temperature median anomaly (°C)",
                "min"
            ),
            "Mean precipitation anomaly (mm/month)": (
                "Precipitation median anomaly (mm/month)",
                "mean"
            ),
            "Max precipitation anomaly (mm/month)": (
                "Precipitation median anomaly (mm/month)",
                "max"
            ),
            "Min precipitation anomaly (mm/month)": (
                "Precipitation median anomaly (mm/month)",
                "min"
            ),
        }
    )
)

summary_numeric_cols = [
    "Mean temperature anomaly (°C)",
    "Max temperature anomaly (°C)",
    "Min temperature anomaly (°C)",
    "Mean precipitation anomaly (mm/month)",
    "Max precipitation anomaly (mm/month)",
    "Min precipitation anomaly (mm/month)",
]

peru_polygon_summary[summary_numeric_cols] = (
    peru_polygon_summary[summary_numeric_cols]
    .round(2)
)

peru_polygon_summary = peru_polygon_summary.sort_values(
    "Mean temperature anomaly (°C)",
    ascending=False
)

print("\nPolygon-based Peru subregional summary across final 3 ECMWF forecast months:")
display(peru_polygon_summary)


# ------------------------------------------------------------
# 17F. Heatmaps: polygon subregions x final 3 forecast months
# ------------------------------------------------------------

subregion_order = list(peru_polygon_summary["Subregion"])

month_order = (
    peru_polygon_forecast_table[["Month_dt", "Month"]]
    .drop_duplicates()
    .sort_values("Month_dt")
)

month_order_labels = month_order["Month"].tolist()

temp_matrix = (
    peru_polygon_forecast_table
    .pivot(
        index="Subregion",
        columns="Month",
        values="Temperature median anomaly (°C)"
    )
    .reindex(index=subregion_order, columns=month_order_labels)
)

precip_matrix = (
    peru_polygon_forecast_table
    .pivot(
        index="Subregion",
        columns="Month",
        values="Precipitation median anomaly (mm/month)"
    )
    .reindex(index=subregion_order, columns=month_order_labels)
)

t_abs = np.nanmax(np.abs(temp_matrix.values))
p_abs = np.nanmax(np.abs(precip_matrix.values))

if not np.isfinite(t_abs) or t_abs == 0:
    t_abs = 1

if not np.isfinite(p_abs) or p_abs == 0:
    p_abs = 1

fig, axes = plt.subplots(
    ncols=2,
    figsize=(13, max(5, 0.55 * len(subregion_order))),
    sharey=True
)

ax_t, ax_p = axes

im_t = ax_t.imshow(
    temp_matrix.values,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-t_abs,
    vmax=t_abs
)

im_p = ax_p.imshow(
    precip_matrix.values,
    aspect="auto",
    cmap="BrBG",
    vmin=-p_abs,
    vmax=p_abs
)

ax_t.set_title("Temperature anomaly")
ax_p.set_title("Precipitation anomaly")

ax_t.set_xticks(np.arange(len(month_order_labels)))
ax_p.set_xticks(np.arange(len(month_order_labels)))

ax_t.set_xticklabels(
    month_order_labels,
    rotation=30,
    ha="right"
)

ax_p.set_xticklabels(
    month_order_labels,
    rotation=30,
    ha="right"
)

ax_t.set_yticks(np.arange(len(subregion_order)))
ax_t.set_yticklabels(subregion_order)

ax_p.set_yticks(np.arange(len(subregion_order)))
ax_p.set_yticklabels(subregion_order)

for ax in axes:
    ax.set_xlabel("Forecast month")

for i in range(temp_matrix.shape[0]):
    for j in range(temp_matrix.shape[1]):
        val = temp_matrix.values[i, j]
        if np.isfinite(val):
            ax_t.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black"
            )

for i in range(precip_matrix.shape[0]):
    for j in range(precip_matrix.shape[1]):
        val = precip_matrix.values[i, j]
        if np.isfinite(val):
            ax_p.text(
                j,
                i,
                f"{val:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black"
            )

cbar_t = fig.colorbar(
    im_t,
    ax=ax_t,
    fraction=0.046,
    pad=0.04
)

cbar_p = fig.colorbar(
    im_p,
    ax=ax_p,
    fraction=0.046,
    pad=0.04
)

cbar_t.set_label("°C anomaly")
cbar_p.set_label("mm/month anomaly")

fig.suptitle(
    "Peru polygon-based ECMWF forecast anomalies: final 3 forecast months",
    y=1.03,
    fontsize=14
)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 17G. Bar chart: final-3-month mean anomalies by polygon subregion
# ------------------------------------------------------------

plot_df = peru_polygon_summary.copy()

x = np.arange(len(plot_df))

fig, axes = plt.subplots(
    ncols=2,
    figsize=(14, 4.8),
    sharex=True
)

ax_t, ax_p = axes

ax_t.bar(
    x,
    plot_df["Mean temperature anomaly (°C)"].values,
    color="firebrick",
    alpha=0.75
)

ax_t.axhline(
    0,
    color="gray",
    linewidth=0.8
)

ax_t.set_title("Mean temperature anomaly")
ax_t.set_ylabel("°C")
ax_t.set_xticks(x)

ax_t.set_xticklabels(
    plot_df["Subregion"],
    rotation=35,
    ha="right"
)

ax_t.grid(
    True,
    axis="y",
    linestyle="--",
    alpha=0.35
)

ax_p.bar(
    x,
    plot_df["Mean precipitation anomaly (mm/month)"].values,
    color="royalblue",
    alpha=0.75
)

ax_p.axhline(
    0,
    color="gray",
    linewidth=0.8
)

ax_p.set_title("Mean precipitation anomaly")
ax_p.set_ylabel("mm/month")
ax_p.set_xticks(x)

ax_p.set_xticklabels(
    plot_df["Subregion"],
    rotation=35,
    ha="right"
)

ax_p.grid(
    True,
    axis="y",
    linestyle="--",
    alpha=0.35
)

fig.suptitle(
    "Peru polygon-based ECMWF forecast comparison: final 3 forecast months",
    y=1.05,
    fontsize=14
)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 17H. Optional diagnostic map of polygon outlines
# ------------------------------------------------------------

if CARTOPY_AVAILABLE:

    proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(
        figsize=(6, 8),
        subplot_kw={"projection": proj}
    )

    ax.set_extent(
        [-84, -66, -21, 1],
        crs=proj
    )

    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    if peru_country_geom is not None:
        ax.add_geometries(
            [peru_country_geom],
            crs=proj,
            facecolor="none",
            edgecolor="black",
            linewidth=1.2
        )

    for name, geom in peru_subregion_polygons.items():
        ax.add_geometries(
            [geom],
            crs=proj,
            facecolor="none",
            edgecolor="red",
            linewidth=1.0,
            alpha=0.8
        )

        centroid = geom.centroid

        ax.text(
            centroid.x,
            centroid.y,
            name,
            transform=proj,
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.65,
                pad=1
            )
        )

    ax.set_title("Approximate Peru polygon subregions used for ECMWF summary")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 17I. Quick interpretation helper
# ------------------------------------------------------------

if len(peru_polygon_summary) > 0:

    warmest_subregion = peru_polygon_summary.iloc[0]["Subregion"]
    warmest_value = peru_polygon_summary.iloc[0]["Mean temperature anomaly (°C)"]

    wettest_subregion = (
        peru_polygon_summary
        .sort_values(
            "Mean precipitation anomaly (mm/month)",
            ascending=False
        )
        .iloc[0]
    )

    driest_subregion = (
        peru_polygon_summary
        .sort_values(
            "Mean precipitation anomaly (mm/month)",
            ascending=True
        )
        .iloc[0]
    )

    print("\nQuick polygon-based Peru interpretation:")
    print(
        f"  Warmest mean temperature anomaly: "
        f"{warmest_subregion} ({warmest_value:.2f} °C)."
    )

    print(
        f"  Wettest mean precipitation anomaly: "
        f"{wettest_subregion['Subregion']} "
        f"({wettest_subregion['Mean precipitation anomaly (mm/month)']:.1f} mm/month)."
    )

    print(
        f"  Driest mean precipitation anomaly: "
        f"{driest_subregion['Subregion']} "
        f"({driest_subregion['Mean precipitation anomaly (mm/month)']:.1f} mm/month)."
    )

print(
    "\nMethod note: this block uses polygon masks and area-weighted means over "
    "forecast grid-cell centres inside each polygon. If Natural Earth Peru boundary "
    "is available, the subregion polygons are clipped to Peru. For a thesis or report, "
    "you can replace the approximate polygons with official Andes/coast/Amazon, "
    "department, watershed, or elevation-zone shapefiles."
)
# ------------------------------------------------------------
# 17J.5 Plot time series by Peru subregion
# Updated version: connects last ERA5 point to first ECMWF forecast point
# ------------------------------------------------------------

def plot_peru_polygon_zone_time_series(
    start_date=PERU_TS_START,
    end_date=PERU_TS_END,
    connect_observed_to_forecast=True
):
    """
    Plot ERA5 observed and ECMWF forecast anomaly time series for each Peru zone.

    One row per subregion, two columns:
      - temperature anomaly
      - precipitation anomaly

    The observed ERA5 line and forecast ECMWF line are separate data products.
    If connect_observed_to_forecast=True, a dashed connector is drawn from the
    final observed value to the first forecast median value.
    """

    subregions = list(peru_polygon_era5_series.keys())
    n_regions = len(subregions)

    fig, axes = plt.subplots(
        nrows=n_regions,
        ncols=2,
        figsize=(15, max(3.0 * n_regions, 6)),
        sharex=False
    )

    if n_regions == 1:
        axes = np.array([axes])

    for i, subregion_name in enumerate(subregions):

        ax_t = axes[i, 0]
        ax_p = axes[i, 1]

        # ----------------------------------------------------
        # ERA5 observed
        # ----------------------------------------------------

        obs_t2 = peru_polygon_era5_series[subregion_name]["t2m"]
        obs_tp = peru_polygon_era5_series[subregion_name]["tp"]

        obs_t2 = standardize_series_monthly_time(obs_t2)
        obs_tp = standardize_series_monthly_time(obs_tp)

        if start_date is not None:
            obs_t2 = obs_t2.sel(time=slice(start_date, None))
            obs_tp = obs_tp.sel(time=slice(start_date, None))

        if end_date is not None:
            obs_t2 = obs_t2.sel(time=slice(None, end_date))
            obs_tp = obs_tp.sel(time=slice(None, end_date))

        obs_t2_time = pd.DatetimeIndex(pd.to_datetime(obs_t2.time.values))
        obs_tp_time = pd.DatetimeIndex(pd.to_datetime(obs_tp.time.values))

        ax_t.plot(
            obs_t2_time,
            obs_t2.values,
            color="dimgray",
            linewidth=1.2,
            label="ERA5 observed"
        )

        ax_p.plot(
            obs_tp_time,
            obs_tp.values,
            color="dimgray",
            linewidth=1.2,
            label="ERA5 observed"
        )

        # ----------------------------------------------------
        # ECMWF forecast
        # ----------------------------------------------------

        if subregion_name in peru_polygon_ecmwf_series:

            fc_t = peru_polygon_ecmwf_series[subregion_name]["temperature_quantiles"]
            fc_p = peru_polygon_ecmwf_series[subregion_name]["precipitation_quantiles"]

            # Make sure forecast time is datetime
            if len(fc_t) > 0:
                fc_t = fc_t.copy()
                fc_t["time"] = pd.DatetimeIndex(
                    pd.to_datetime(fc_t["time"])
                ).to_period("M").to_timestamp()

            if len(fc_p) > 0:
                fc_p = fc_p.copy()
                fc_p["time"] = pd.DatetimeIndex(
                    pd.to_datetime(fc_p["time"])
                ).to_period("M").to_timestamp()

            # ------------------------------------------------
            # Temperature connector
            # ------------------------------------------------

            if connect_observed_to_forecast and len(fc_t) > 0 and len(obs_t2.time) > 0:

                last_obs_t_time = pd.Timestamp(obs_t2.time.values[-1]).to_period("M").to_timestamp()
                last_obs_t_val = float(obs_t2.isel(time=-1).values)

                first_fc_t_time = pd.Timestamp(fc_t["time"].iloc[0]).to_period("M").to_timestamp()
                first_fc_t_val = float(fc_t["median"].iloc[0])

                ax_t.plot(
                    [last_obs_t_time, first_fc_t_time],
                    [last_obs_t_val, first_fc_t_val],
                    color="firebrick",
                    linestyle="--",
                    linewidth=2.2,
                    alpha=0.8,
                    label="ERA5 to forecast transition"
                )

            # ------------------------------------------------
            # Precipitation connector
            # ------------------------------------------------

            if connect_observed_to_forecast and len(fc_p) > 0 and len(obs_tp.time) > 0:

                last_obs_p_time = pd.Timestamp(obs_tp.time.values[-1]).to_period("M").to_timestamp()
                last_obs_p_val = float(obs_tp.isel(time=-1).values)

                first_fc_p_time = pd.Timestamp(fc_p["time"].iloc[0]).to_period("M").to_timestamp()
                first_fc_p_val = float(fc_p["median"].iloc[0])

                ax_p.plot(
                    [last_obs_p_time, first_fc_p_time],
                    [last_obs_p_val, first_fc_p_val],
                    color="royalblue",
                    linestyle="--",
                    linewidth=2.2,
                    alpha=0.8,
                    label="ERA5 to forecast transition"
                )

            # ------------------------------------------------
            # Temperature forecast median and spread
            # ------------------------------------------------

            if len(fc_t) > 0:

                if SHOW_FORECAST_5_95:
                    ax_t.fill_between(
                        fc_t["time"],
                        fc_t["p05"],
                        fc_t["p95"],
                        color="firebrick",
                        alpha=0.10,
                        label=f"{FORECAST_SHORT_LABEL} 5-95%"
                    )

                if SHOW_FORECAST_IQR:
                    ax_t.fill_between(
                        fc_t["time"],
                        fc_t["p25"],
                        fc_t["p75"],
                        color="firebrick",
                        alpha=0.20,
                        label=f"{FORECAST_SHORT_LABEL} IQR"
                    )

                ax_t.plot(
                    fc_t["time"],
                    fc_t["median"],
                    color="firebrick",
                    linestyle="--",
                    linewidth=2.2,
                    label=f"{FORECAST_SHORT_LABEL} median"
                )

            # ------------------------------------------------
            # Precipitation forecast median and spread
            # ------------------------------------------------

            if len(fc_p) > 0:

                if SHOW_FORECAST_5_95:
                    ax_p.fill_between(
                        fc_p["time"],
                        fc_p["p05"],
                        fc_p["p95"],
                        color="royalblue",
                        alpha=0.10,
                        label=f"{FORECAST_SHORT_LABEL} 5-95%"
                    )

                if SHOW_FORECAST_IQR:
                    ax_p.fill_between(
                        fc_p["time"],
                        fc_p["p25"],
                        fc_p["p75"],
                        color="royalblue",
                        alpha=0.20,
                        label=f"{FORECAST_SHORT_LABEL} IQR"
                    )

                ax_p.plot(
                    fc_p["time"],
                    fc_p["median"],
                    color="royalblue",
                    linestyle="--",
                    linewidth=2.2,
                    label=f"{FORECAST_SHORT_LABEL} median"
                )

            # ------------------------------------------------
            # Print check for observed and forecast dates
            # ------------------------------------------------

            if len(fc_t) > 0 and len(obs_t2.time) > 0:
                print(
                    subregion_name,
                    "| temperature:",
                    "last ERA5 obs =",
                    pd.Timestamp(obs_t2.time.values[-1]).strftime("%Y-%m"),
                    "| first forecast =",
                    pd.Timestamp(fc_t["time"].iloc[0]).strftime("%Y-%m")
                )

            if len(fc_p) > 0 and len(obs_tp.time) > 0:
                print(
                    subregion_name,
                    "| precipitation:",
                    "last ERA5 obs =",
                    pd.Timestamp(obs_tp.time.values[-1]).strftime("%Y-%m"),
                    "| first forecast =",
                    pd.Timestamp(fc_p["time"].iloc[0]).strftime("%Y-%m")
                )

        # ----------------------------------------------------
        # Styling
        # ----------------------------------------------------

        ax_t.axhline(
            0,
            color="black",
            linewidth=0.8,
            alpha=0.6
        )

        ax_p.axhline(
            0,
            color="black",
            linewidth=0.8,
            alpha=0.6
        )

        ax_t.set_title(f"{subregion_name} - temperature")
        ax_p.set_title(f"{subregion_name} - precipitation")

        ax_t.set_ylabel("°C anomaly")
        ax_p.set_ylabel("mm/month anomaly")

        for ax in [ax_t, ax_p]:

            ax.grid(
                True,
                linestyle="--",
                alpha=0.35
            )

            # Remove duplicate legend labels
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))

            ax.legend(
                unique.values(),
                unique.keys(),
                fontsize=8,
                loc="best"
            )

    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")

    fig.suptitle(
        "Peru polygon-zone anomaly time series: ERA5 observed and ECMWF forecast",
        y=1.002,
        fontsize=15
    )

    plt.tight_layout()
    plt.show()


plot_peru_polygon_zone_time_series(
    start_date=PERU_TS_START,
    end_date=PERU_TS_END,
    connect_observed_to_forecast=True
)
