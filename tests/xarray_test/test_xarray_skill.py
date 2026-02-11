"""
Test script that exercises xarray features using ONLY patterns from the skill file:
  /Users/ian/Documents/dev/scientific-python-skills/skills/xarray.md

Each section corresponds to a specific skill-file topic. We use a temp directory
for all file I/O and print clear pass/fail markers.
"""

import tempfile
import os
import traceback
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def ok(msg):
    print(f"  [PASS] {msg}")


def fail(msg, exc=None):
    print(f"  [FAIL] {msg}")
    if exc:
        traceback.print_exc()


# ===========================================================================
# 1. Create a Dataset with realistic climate-like data
#    Skill-file pattern: xr.Dataset(...), xr.date_range(..., unit="s")
# ===========================================================================
section("1. Create a Dataset with realistic climate-like data")
try:
    import xarray as xr

    # Skill file: xr.date_range("2000-01-01", periods=365, freq="D", unit="s")
    times = xr.date_range("2000-01-01", periods=365, freq="D", unit="s")

    lats = np.linspace(-89, 89, 90)
    lons = np.linspace(-179, 179, 180)

    rng = np.random.default_rng(42)
    temperature = rng.standard_normal((365, 90, 180)).astype("float32") * 10 + 280
    pressure = rng.standard_normal((365, 90, 180)).astype("float32") * 50 + 101325

    # Skill file pattern: xr.Dataset({name: (dims, data)}, coords={...})
    ds = xr.Dataset(
        {
            "temperature": (("time", "lat", "lon"), temperature, {"units": "K"}),
            "pressure": (("time", "lat", "lon"), pressure, {"units": "Pa"}),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )
    print(f"  Dataset created: {ds.dims}")
    assert ds.dims == {"time": 365, "lat": 90, "lon": 180}
    ok("Dataset creation with xr.date_range(unit='s')")
except Exception as e:
    fail("Dataset creation", e)


# ===========================================================================
# 2. apply_ufunc with dask
#    Skill-file patterns: chunk(), apply_ufunc with dask='parallelized',
#    output_dtypes, input_core_dims, output_core_dims, core dims -> last axis
# ===========================================================================
section("2. apply_ufunc with dask (standardize time series)")
try:
    # Skill file: ds.chunk({"time": -1, "x": 100})
    ds_chunked = ds.chunk({"time": -1, "lat": 30, "lon": 60})

    # Skill file: core dims are moved to LAST axis (-1)
    def standardize(arr):
        """Standardize along last axis (core dim = time)."""
        mean = np.nanmean(arr, axis=-1, keepdims=True)
        std = np.nanstd(arr, axis=-1, keepdims=True)
        return (arr - mean) / std

    # Skill file pattern: apply_ufunc with dask='parallelized' + output_dtypes
    result = xr.apply_ufunc(
        standardize,
        ds_chunked["temperature"],
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        dask="parallelized",
        output_dtypes=[float],
    )

    # Trigger computation
    computed = result.compute()
    assert computed.shape == (90, 180, 365)  # core dim (time) moved to last
    # Actually check the values make sense -- mean near 0, std near 1
    sample_mean = float(computed.isel(lat=0, lon=0).mean())
    sample_std = float(computed.isel(lat=0, lon=0).std())
    assert abs(sample_mean) < 0.01, f"mean = {sample_mean}"
    assert abs(sample_std - 1.0) < 0.01, f"std = {sample_std}"
    ok("apply_ufunc with dask='parallelized', input/output_core_dims, output_dtypes")
except Exception as e:
    fail("apply_ufunc with dask", e)


# ===========================================================================
# 3. GroupBy with resample -- monthly means, seasonal anomalies
#    Skill-file patterns: resample(time="ME").mean(), groupby("time.season")
#    Note: "ME" not "M", "YE" not "Y"
# ===========================================================================
section("3. GroupBy with resample (monthly means, seasonal anomalies)")
try:
    # Skill file: ds.resample(time="ME").mean()  -- "ME" = month-end frequency
    monthly = ds.resample(time="ME").mean()
    print(f"  Monthly means: {monthly.dims}")
    assert monthly.dims["time"] == 12  # 365 days in 2000 -> 12 months
    ok("resample(time='ME').mean() produced 12 monthly means")

    # Skill file: ds.groupby("time.season").map(lambda x: x - x.mean("time"))
    seasonal_anomaly = ds.groupby("time.season").map(lambda x: x - x.mean("time"))
    assert seasonal_anomaly.dims == ds.dims
    ok("groupby('time.season').map() seasonal anomalies")

    # Skill file: ds.groupby("time.month").mean()
    monthly_clim = ds.groupby("time.month").mean()
    assert "month" in monthly_clim.dims
    ok("groupby('time.month').mean() monthly climatology")

except Exception as e:
    fail("GroupBy / Resample", e)


# ===========================================================================
# 4. Write to zarr v3
#    Skill-file patterns: zarr_format=3, BloscCodec, compressors (plural, tuple)
# ===========================================================================
section("4. Write to zarr v3 with encoding")
try:
    from zarr.codecs import BloscCodec

    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = os.path.join(tmpdir, "climate_v3.zarr")

        # Skill file pattern: ds.to_zarr("store.zarr", zarr_format=3)
        # Skill file pattern: encoding={"var": {"compressors": (compressor,)}}
        ds_small = ds.isel(time=slice(0, 10))  # smaller for speed
        ds_small.to_zarr(
            zarr_path,
            zarr_format=3,
            encoding={
                "temperature": {"compressors": (compressor,)},
                "pressure": {"compressors": (compressor,)},
            },
        )

        # Read back
        ds_read = xr.open_zarr(zarr_path)
        xr.testing.assert_equal(ds_small.load(), ds_read.load())
        ok("to_zarr(zarr_format=3) with BloscCodec compressors=(codec,)")

except Exception as e:
    fail("Zarr v3 write", e)


# ===========================================================================
# 5. open_mfdataset pattern
#    Skill-file patterns: write to multiple NetCDF files, read back with
#    open_mfdataset(..., compat='override', coords='minimal')
# ===========================================================================
section("5. open_mfdataset pattern")
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Split dataset into 3 time chunks and write as separate NetCDF files
        chunk_size = 365 // 3  # ~121 days each
        paths = []
        for i in range(3):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < 2 else 365
            chunk_ds = ds.isel(time=slice(start, end))
            path = os.path.join(tmpdir, f"data_{i:02d}.nc")
            chunk_ds.to_netcdf(path)
            paths.append(path)
            print(f"  Wrote {path} with time range [{start}:{end}]")

        # Skill file: xr.open_mfdataset('data/*.nc', compat='override',
        #             coords='minimal', data_vars='minimal')
        ds_multi = xr.open_mfdataset(
            os.path.join(tmpdir, "data_*.nc"),
            compat="override",
            coords="minimal",
            data_vars="minimal",
        )
        assert ds_multi.dims["time"] == 365
        ok("open_mfdataset with compat='override', coords='minimal'")

        # Clean up
        ds_multi.close()

except Exception as e:
    fail("open_mfdataset", e)


# ===========================================================================
# 6. Encoding gotchas
#    Skill-file patterns: drop_encoding(), encoding persists through operations
# ===========================================================================
section("6. Encoding gotchas (drop_encoding before re-saving)")
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path_in = os.path.join(tmpdir, "input.zarr")
        zarr_path_out = os.path.join(tmpdir, "output.zarr")

        # Write original dataset (small) to zarr
        ds_small = ds.isel(time=slice(0, 30))
        ds_small.to_zarr(zarr_path_in)

        # Read back -- encoding is attached
        ds_from_zarr = xr.open_zarr(zarr_path_in)
        print(f"  Encoding on 'temperature': {ds_from_zarr['temperature'].encoding}")

        # Skill file: GOTCHA -- encoding persists and can cause errors
        # Take a subset (potentially incompatible chunks)
        ds_subset = ds_from_zarr.isel(time=slice(0, 10))

        # Skill file pattern: ds_subset.drop_encoding()
        ds_subset = ds_subset.drop_encoding()
        print(f"  After drop_encoding: {ds_subset['temperature'].encoding}")
        assert ds_subset["temperature"].encoding == {}
        ok("drop_encoding() clears encoding dict")

        # Now write -- should succeed
        ds_subset.to_zarr(zarr_path_out)
        ok("Writing after drop_encoding() succeeds")

except Exception as e:
    fail("Encoding gotchas", e)


# ===========================================================================
# 7. DataTree
#    Skill-file patterns: xr.DataTree, DataTree.from_dict, navigation,
#    subtree_with_keys, map_over_datasets
# ===========================================================================
section("7. DataTree")
try:
    # Skill file: xr.DataTree.from_dict({"/": ds, "/group1": ds1, ...})
    ds_root = xr.Dataset({"global_temp": ("x", [1.0, 2.0, 3.0])})
    ds_surface = xr.Dataset({
        "temp_2m": (("lat", "lon"), rng.standard_normal((10, 20)).astype("float32")),
    })
    ds_upper = xr.Dataset({
        "temp_500hpa": (("lat", "lon"), rng.standard_normal((10, 20)).astype("float32")),
    })

    dt = xr.DataTree.from_dict({
        "/": ds_root,
        "/surface": ds_surface,
        "/upper_air": ds_upper,
    })
    print(f"  DataTree: {dt}")

    # Skill file: dt["/group/subgroup"]
    surface_node = dt["/surface"]
    assert "temp_2m" in surface_node.dataset.data_vars
    ok("DataTree.from_dict and navigation with dt['/surface']")

    # Skill file: dt.children
    children = dt.children
    assert "surface" in children
    assert "upper_air" in children
    ok("dt.children access")

    # Skill file: for path, node in dt.subtree_with_keys:
    paths_seen = []
    for path, node in dt.subtree_with_keys:
        paths_seen.append(str(path))
    print(f"  Paths in subtree: {paths_seen}")
    ok("dt.subtree_with_keys iteration")

    # Skill file: dt.map_over_datasets(lambda ds: ds.mean())
    result_tree = dt.map_over_datasets(lambda ds: ds.mean())
    ok("map_over_datasets(lambda ds: ds.mean())")

except Exception as e:
    fail("DataTree", e)


# ===========================================================================
# 8. Time series with CFDatetimeCoder
#    Skill-file patterns: xr.coders.CFDatetimeCoder(time_unit="s"),
#    pass coder to decode_times=
# ===========================================================================
section("8. Time series with CFDatetimeCoder")
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        nc_path = os.path.join(tmpdir, "time_test.nc")

        # Create a dataset with time encoded manually
        ds_time = xr.Dataset(
            {"temp": (("time",), rng.standard_normal(100).astype("float32"))},
            coords={"time": xr.date_range("2000-01-01", periods=100, freq="D", unit="s")},
        )
        # Write with specific time encoding
        ds_time.to_netcdf(nc_path, encoding={
            "time": {"units": "days since 2000-01-01", "dtype": "float64"},
        })

        # Skill file pattern: xr.coders.CFDatetimeCoder(time_unit="s")
        coder = xr.coders.CFDatetimeCoder(time_unit="s")

        # Skill file pattern: xr.open_dataset("file.nc", decode_times=coder)
        ds_decoded = xr.open_dataset(nc_path, decode_times=coder)
        print(f"  Time dtype: {ds_decoded.time.dtype}")
        print(f"  Time values (first 3): {ds_decoded.time.values[:3]}")

        # Verify it is second-resolution
        assert "datetime64" in str(ds_decoded.time.dtype)
        ok("CFDatetimeCoder(time_unit='s') with decode_times=coder")

        ds_decoded.close()

except Exception as e:
    fail("CFDatetimeCoder", e)


# ===========================================================================
# Summary
# ===========================================================================
section("ALL SECTIONS COMPLETE")
print("  Review output above for [PASS] / [FAIL] markers.\n")
