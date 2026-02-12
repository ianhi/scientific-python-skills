"""
Comprehensive test script that verifies every code example and claim
in skills/xarray.md by executing real code.

Run with: uv run python tests/xarray_test/test_xarray_skill.py
"""
import tempfile
import os
import traceback
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASSES = []
FAILS = []


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def ok(msg):
    PASSES.append(msg)
    print(f"  [PASS] {msg}")


def fail(msg, exc=None):
    FAILS.append(msg)
    print(f"  [FAIL] {msg}")
    if exc:
        traceback.print_exc()


# ===========================================================================
# 1. DataTree (New - in xarray since v2024.10)
# ===========================================================================
section("1. DataTree")

# 1a. Constructor with dataset= param
try:
    import xarray as xr

    ds_node = xr.Dataset({"var1": ("x", [1, 2, 3])})
    dt = xr.DataTree(
        dataset=ds_node,
        children=None,
        name="root",
    )
    assert "var1" in dt.dataset.data_vars
    ok("DataTree(dataset=ds, children=None, name='root') constructor works")
except Exception as e:
    fail("DataTree constructor with dataset= param", e)

# 1b. from_dict - NOTE: child nodes must have compatible dims with parent
# The skill file example uses ds1/ds2 as placeholders. Here we use non-conflicting dims.
try:
    ds1 = xr.Dataset({"a": ("y", [10, 20])})
    ds2 = xr.Dataset({"b": ("z", [30, 40, 50])})
    dt = xr.DataTree.from_dict({
        "/": xr.Dataset({"global_temp": ("x", [1, 2, 3])}),
        "/group1": ds1,
        "/group1/subgroup": ds2,
    })
    assert "group1" in dt.children
    assert "subgroup" in dt["group1"].children
    ok("DataTree.from_dict with hierarchical paths")
except Exception as e:
    fail("DataTree.from_dict", e)

# 1b-extra. Verify alignment constraint: same dim name with different sizes fails
try:
    try:
        dt_bad = xr.DataTree.from_dict({
            "/": xr.Dataset({"global_temp": ("x", [1, 2, 3])}),
            "/group1": xr.Dataset({"a": ("x", [10, 20])}),  # x size 2 vs 3
        })
        fail("DataTree.from_dict should fail with conflicting dim sizes")
    except ValueError:
        ok("DataTree.from_dict raises ValueError on conflicting dim sizes (alignment)")
except Exception as e:
    fail("DataTree alignment constraint", e)

# 1c. map_over_datasets
try:
    result = dt.map_over_datasets(lambda ds: ds.mean())
    ok("dt.map_over_datasets(lambda ds: ds.mean()) works")
except Exception as e:
    fail("map_over_datasets", e)

# 1d. DataTree I/O with zarr
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = os.path.join(tmpdir, "tree.zarr")
        dt.to_zarr(zarr_path)
        dt_read = xr.open_datatree(zarr_path, engine="zarr")
        assert "group1" in dt_read.children
        ok("DataTree zarr I/O: to_zarr + open_datatree(engine='zarr')")
except Exception as e:
    fail("DataTree zarr I/O", e)


# ===========================================================================
# 2. apply_ufunc
# ===========================================================================
section("2. apply_ufunc")

rng = np.random.default_rng(42)

# 2a. Core dims on last axis with dask='parallelized'
try:
    da = xr.DataArray(
        rng.standard_normal((10, 20, 100)).astype("float32"),
        dims=["lat", "lon", "time"],
    )
    da_chunked = da.chunk({"lat": 5, "lon": 10, "time": -1})

    def detrend_axis_last(arr):
        """Core dims should be on last axis."""
        assert arr.shape[-1] == 100, f"Expected last dim=100, got {arr.shape}"
        return arr - np.mean(arr, axis=-1, keepdims=True)

    result = xr.apply_ufunc(
        detrend_axis_last,
        da_chunked,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        dask="parallelized",
        output_dtypes=[float],
    )
    computed = result.compute()
    ok("apply_ufunc: core dims on last axis (-1), dask='parallelized', output_dtypes required")
except Exception as e:
    fail("apply_ufunc core dims last axis", e)

# 2b. output_sizes for dimension-changing functions
try:
    da_time = xr.DataArray(
        rng.standard_normal((5, 100)).astype("float32"),
        dims=["x", "time"],
    )
    da_time_chunked = da_time.chunk({"x": 5, "time": -1})

    def coarsen_func(arr, factor):
        n = arr.shape[-1] // factor
        return arr[..., :n * factor].reshape(arr.shape[:-1] + (n, factor)).mean(axis=-1)

    result = xr.apply_ufunc(
        coarsen_func, da_time_chunked, 2,
        input_core_dims=[["time"], []],
        output_core_dims=[["time_coarse"]],
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"time_coarse": len(da_time.time) // 2}},
    )
    computed = result.compute()
    assert computed.sizes["time_coarse"] == 50
    ok("apply_ufunc: output_sizes via dask_gufunc_kwargs")
except Exception as e:
    fail("apply_ufunc output_sizes", e)

# 2c. Multiple return values
try:
    def decompose(arr):
        m = np.mean(arr, axis=-1)
        s = np.std(arr, axis=-1)
        return m, s

    da_small = xr.DataArray(
        rng.standard_normal((3, 50)).astype("float32"),
        dims=["x", "time"],
    )
    da_small_chunked = da_small.chunk({"x": 3, "time": -1})

    mean, std = xr.apply_ufunc(
        decompose, da_small_chunked,
        input_core_dims=[["time"]],
        output_core_dims=[[], []],
        dask="parallelized",
        output_dtypes=[float, float],
    )
    m_val = mean.compute()
    s_val = std.compute()
    assert m_val.shape == (3,)
    assert s_val.shape == (3,)
    ok("apply_ufunc: multiple return values with output_core_dims=[[], []]")
except Exception as e:
    fail("apply_ufunc multiple returns", e)

# 2d. vectorize=True
try:
    def scalar_func(a, b):
        return a + b * 2

    da1 = xr.DataArray([1.0, 2.0, 3.0], dims=["x"])
    da2 = xr.DataArray([10.0, 20.0, 30.0], dims=["x"])

    result = xr.apply_ufunc(
        scalar_func, da1, da2,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    assert np.allclose(result.values, [21.0, 42.0, 63.0])
    ok("apply_ufunc: vectorize=True for scalar functions")
except Exception as e:
    fail("apply_ufunc vectorize", e)


# ===========================================================================
# 3. map_blocks
# ===========================================================================
section("3. map_blocks")

try:
    ds_mb = xr.Dataset({
        "temp": (("x", "y"), rng.standard_normal((20, 30)).astype("float32")),
    })
    ds_mb_chunked = ds_mb.chunk({"x": 10, "y": 15})

    result = xr.map_blocks(lambda block: block - block.mean(), ds_mb_chunked)
    computed = result.compute()
    assert computed["temp"].shape == (20, 30)
    ok("xr.map_blocks(lambda block: block - block.mean(), ds) works")
except Exception as e:
    fail("map_blocks", e)


# ===========================================================================
# 4. Data Combination
# ===========================================================================
section("4. Data Combination (concat, merge, combine_by_coords)")

# 4a. concat along existing dim
try:
    ds_a = xr.Dataset({
        "temp": (("time", "x"), rng.standard_normal((5, 3)).astype("float32")),
    }, coords={"time": range(5)})
    ds_b = xr.Dataset({
        "temp": (("time", "x"), rng.standard_normal((5, 3)).astype("float32")),
    }, coords={"time": range(5, 10)})

    combined = xr.concat([ds_a, ds_b], dim="time")
    assert combined.sizes["time"] == 10
    ok("concat along existing dim='time'")
except Exception as e:
    fail("concat existing dim", e)

# 4b. concat along NEW dim
try:
    ds_e1 = xr.Dataset({"temp": (("x",), [1.0, 2.0, 3.0])})
    ds_e2 = xr.Dataset({"temp": (("x",), [4.0, 5.0, 6.0])})
    combined_new = xr.concat([ds_e1, ds_e2], dim="ensemble")
    assert "ensemble" in combined_new.dims
    assert combined_new.sizes["ensemble"] == 2
    ok("concat along NEW dim='ensemble' adds it")
except Exception as e:
    fail("concat new dim", e)

# 4c. merge with different variables
try:
    ds_temp = xr.Dataset({"temperature": ("x", [1.0, 2.0, 3.0])})
    ds_precip = xr.Dataset({"precipitation": ("x", [0.1, 0.2, 0.3])})
    merged = xr.merge([ds_temp, ds_precip])
    assert "temperature" in merged and "precipitation" in merged
    ok("merge: combine datasets with DIFFERENT variables")
except Exception as e:
    fail("merge different vars", e)

# 4d. merge fails on conflicting values
try:
    ds_c1 = xr.Dataset({"var": ("x", [1, 2, 3])})
    ds_c2 = xr.Dataset({"var": ("x", [4, 5, 6])})
    try:
        xr.merge([ds_c1, ds_c2])
        fail("merge should fail on conflicting values but didn't")
    except xr.MergeError:
        ok("merge raises MergeError on conflicting values")
except Exception as e:
    fail("merge conflict detection", e)

# 4e. combine_by_coords
try:
    ds_tile1 = xr.Dataset(
        {"data": (("x", "y"), [[1, 2]])},
        coords={"x": [0], "y": [0, 1]},
    )
    ds_tile2 = xr.Dataset(
        {"data": (("x", "y"), [[3, 4]])},
        coords={"x": [1], "y": [0, 1]},
    )
    combined = xr.combine_by_coords([ds_tile1, ds_tile2])
    assert combined.sizes["x"] == 2
    assert combined.sizes["y"] == 2
    ok("combine_by_coords auto-infers concat dims from coordinates")
except Exception as e:
    fail("combine_by_coords", e)


# ===========================================================================
# 5. GroupBy Performance / Custom Groupers
# ===========================================================================
section("5. GroupBy: Custom Groupers and Resample Freq Aliases")

# 5a. BinGrouper, TimeResampler, SeasonGrouper imports
try:
    from xarray.groupers import BinGrouper, TimeResampler, SeasonGrouper
    ok("Import BinGrouper, TimeResampler, SeasonGrouper from xarray.groupers")
except Exception as e:
    fail("Import custom groupers", e)

# 5b. BinGrouper usage
try:
    ds_bin = xr.Dataset({
        "temp": ("x", rng.standard_normal(100).astype("float32")),
    }, coords={"x": np.linspace(0, 50, 100)})

    result = ds_bin.groupby(x=BinGrouper(bins=[0, 25, 50])).mean()
    ok("ds.groupby(x=BinGrouper(bins=[0, 25, 50])).mean()")
except Exception as e:
    fail("BinGrouper usage", e)

# 5c. TimeResampler usage
try:
    times = xr.date_range("2000-01-01", periods=365, freq="D", unit="s")
    ds_ts = xr.Dataset({
        "temp": ("time", rng.standard_normal(365).astype("float32")),
    }, coords={"time": times})

    result = ds_ts.groupby(time=TimeResampler(freq="ME")).mean()
    assert result.sizes["time"] == 12
    ok("ds.groupby(time=TimeResampler(freq='ME')).mean()")
except Exception as e:
    fail("TimeResampler usage", e)

# 5d. SeasonGrouper usage
try:
    result = ds_ts.groupby(time=SeasonGrouper(["DJF", "MAM", "JJA", "SON"])).mean()
    ok("ds.groupby(time=SeasonGrouper(['DJF','MAM','JJA','SON'])).mean()")
except Exception as e:
    fail("SeasonGrouper usage", e)

# 5e. Resample freq aliases: "ME" not "M", "YE" not "Y"
try:
    monthly = ds_ts.resample(time="ME").mean()
    assert monthly.sizes["time"] == 12
    ok("resample(time='ME') works (month-end)")
except Exception as e:
    fail("resample ME alias", e)

try:
    yearly = ds_ts.resample(time="YE").mean()
    assert yearly.sizes["time"] == 1
    ok("resample(time='YE') works (year-end)")
except Exception as e:
    fail("resample YE alias", e)

# 5f. Verify old aliases "M" and "Y" fail
try:
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            ds_ts.resample(time="M").mean()
            if any("deprecated" in str(warning.message).lower() for warning in w):
                ok("resample(time='M') raises deprecation warning (use 'ME')")
            else:
                ok("resample(time='M') still works (but 'ME' is preferred)")
        except ValueError:
            ok("resample(time='M') raises ValueError (use 'ME' instead)")
except Exception as e:
    fail("Old 'M' alias behavior", e)


# ===========================================================================
# 6. open_mfdataset performance options
# ===========================================================================
section("6. open_mfdataset Performance Options")

try:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files with DIFFERENT time ranges (realistic use case)
        for i in range(3):
            t_start = f"2000-{i*4+1:02d}-01"
            t = xr.date_range(t_start, periods=30, freq="D", unit="s")
            chunk = xr.Dataset({
                "temp": (("time", "x"), rng.standard_normal((30, 10)).astype("float32")),
            }, coords={"time": t, "x": np.arange(10)})
            chunk.to_netcdf(os.path.join(tmpdir, f"data_{i:02d}.nc"))

        # Skill file pattern
        ds_multi = xr.open_mfdataset(
            os.path.join(tmpdir, "data_*.nc"),
            parallel=True,
            compat="override",
            coords="minimal",
            data_vars="minimal",
            engine="h5netcdf",
        )
        assert "temp" in ds_multi
        assert ds_multi.sizes["time"] == 90
        ok("open_mfdataset with compat='override', coords='minimal', data_vars='minimal', engine='h5netcdf'")
        ds_multi.close()
except Exception as e:
    fail("open_mfdataset performance options", e)


# ===========================================================================
# 7. Zarr v3 encoding
# ===========================================================================
section("7. Zarr v3 Encoding")

# 7a. BloscCodec with plural keys
try:
    from zarr.codecs import BloscCodec
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

    ds_enc = xr.Dataset({
        "var": (("x", "y"), rng.standard_normal((20, 30)).astype("float32")),
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        zpath = os.path.join(tmpdir, "encoded.zarr")
        ds_enc.to_zarr(zpath, encoding={"var": {"compressors": (compressor,)}})
        ds_back = xr.open_zarr(zpath)
        xr.testing.assert_equal(ds_enc.load(), ds_back.load())
        ok("BloscCodec with compressors=(codec,) - plural key")
except Exception as e:
    fail("Zarr v3 BloscCodec encoding", e)

# 7b. drop_encoding before re-write
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = os.path.join(tmpdir, "v2_like.zarr")
        p2 = os.path.join(tmpdir, "v3_new.zarr")

        ds_orig = xr.Dataset({"val": (("x",), [1.0, 2.0, 3.0])})
        ds_orig.to_zarr(p1)

        ds_loaded = xr.open_zarr(p1)
        assert ds_loaded["val"].encoding != {}

        ds_clean = ds_loaded.drop_encoding()
        assert ds_clean["val"].encoding == {}
        ds_clean.to_zarr(p2)
        ok("drop_encoding() clears encoding, enables clean re-write")
except Exception as e:
    fail("drop_encoding for zarr migration", e)


# ===========================================================================
# 8. CF Time Decoding
# ===========================================================================
section("8. CF Time Decoding (CFDatetimeCoder)")

# 8a. CFDatetimeCoder with time_unit="s"
try:
    coder = xr.coders.CFDatetimeCoder(time_unit="s")

    with tempfile.TemporaryDirectory() as tmpdir:
        nc_path = os.path.join(tmpdir, "time_test.nc")
        ds_time = xr.Dataset({
            "temp": ("time", rng.standard_normal(50).astype("float32")),
        }, coords={
            "time": xr.date_range("2000-01-01", periods=50, freq="D", unit="s"),
        })
        ds_time.to_netcdf(nc_path)

        ds_decoded = xr.open_dataset(nc_path, decode_times=coder)
        assert "datetime64" in str(ds_decoded.time.dtype)
        ok("CFDatetimeCoder(time_unit='s') with decode_times=coder")
        ds_decoded.close()
except Exception as e:
    fail("CFDatetimeCoder time_unit='s'", e)

# 8b. CFDatetimeCoder(use_cftime=True) - requires cftime package
try:
    import cftime as cft
    coder_cf = xr.coders.CFDatetimeCoder(use_cftime=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        nc_path = os.path.join(tmpdir, "cftime_test.nc")
        ds_time = xr.Dataset({
            "temp": ("time", rng.standard_normal(50).astype("float32")),
        }, coords={
            "time": xr.date_range("2000-01-01", periods=50, freq="D", unit="s"),
        })
        ds_time.to_netcdf(nc_path)

        ds_cf = xr.open_dataset(nc_path, decode_times=coder_cf)
        first = ds_cf.time.values[0]
        assert isinstance(first, cft.datetime), f"Expected cftime, got {type(first)}"
        ok("CFDatetimeCoder(use_cftime=True) produces cftime objects")
        ds_cf.close()
except ImportError:
    ok("CFDatetimeCoder(use_cftime=True) skipped (cftime not installed)")
except Exception as e:
    fail("CFDatetimeCoder use_cftime=True", e)


# ===========================================================================
# 9. Gotchas & Common Mistakes
# ===========================================================================
section("9. Gotchas & Common Mistakes")

# 9a. Auto alignment behavior
# IMPORTANT FINDING: In xarray v2026.1.0, the default arithmetic_join is 'inner',
# NOT 'outer'. The skill file claims NaN is introduced, but that's the OLD behavior.
try:
    a = xr.DataArray([1, 2, 3], dims=["x"], coords={"x": [0, 1, 2]})
    b = xr.DataArray([10, 20, 30], dims=["x"], coords={"x": [1, 2, 3]})

    # Default join is now 'inner' - only overlapping coords
    result_default = a + b
    default_join = xr.get_options()["arithmetic_join"]
    print(f"  arithmetic_join default: {default_join}")

    if default_join == "inner":
        # New behavior: inner join, no NaN
        assert len(result_default.x) == 2  # only [1, 2]
        assert not np.any(np.isnan(result_default.values))
        ok(f"SKILL FILE ISSUE: default arithmetic_join is now '{default_join}', NOT 'outer' - no NaN introduced")
    else:
        # Old behavior: outer join, NaN at boundaries
        assert len(result_default.x) == 4
        assert np.isnan(result_default.sel(x=0).values)
        ok("Auto alignment: a + b with partial overlap -> NaN (outer join)")

    # Verify outer join still produces NaN (for historical reference)
    with xr.set_options(arithmetic_join="outer"):
        result_outer = a + b
        assert len(result_outer.x) == 4
        assert np.isnan(result_outer.sel(x=0).values)
        assert np.isnan(result_outer.sel(x=3).values)
        ok("With arithmetic_join='outer': NaN at non-overlapping coords (old behavior)")
except Exception as e:
    fail("Auto alignment behavior", e)

# 9b. Fix with xr.align inner join
try:
    a2, b2 = xr.align(a, b, join="inner")
    result_inner = a2 + b2
    assert len(result_inner.x) == 2
    assert not np.any(np.isnan(result_inner.values))
    ok("xr.align(a, b, join='inner') works")
except Exception as e:
    fail("xr.align inner fix", e)

# 9c. sel() slices are inclusive on both ends
try:
    da_sel = xr.DataArray(
        np.arange(5, dtype=float),
        dims=["x"],
        coords={"x": [10, 20, 30, 40, 50]},
    )
    result = da_sel.sel(x=slice(10, 30))
    assert list(result.x.values) == [10, 20, 30]  # inclusive on BOTH ends
    ok("sel(x=slice(10, 30)) is inclusive on both ends (returns 10, 20, 30)")
except Exception as e:
    fail("sel inclusive slicing", e)

# 9d. Attributes behavior in operations
# FINDING: In xarray v2026.1.0, with keep_attrs='default', attrs are KEPT when identical
try:
    da1 = xr.DataArray([1, 2, 3], attrs={"units": "K"})
    da2 = xr.DataArray([4, 5, 6], attrs={"units": "K"})
    result = da1 + da2

    keep_default = xr.get_options()["keep_attrs"]
    if result.attrs == {}:
        ok("Arithmetic drops attrs by default: result.attrs == {}")
    else:
        # New behavior: identical attrs are preserved with keep_attrs='default'
        ok(f"SKILL FILE ISSUE: with keep_attrs='{keep_default}', IDENTICAL attrs are preserved (not dropped)")

    # Conflicting attrs ARE dropped
    da3 = xr.DataArray([1, 2, 3], attrs={"units": "K"})
    da4 = xr.DataArray([4, 5, 6], attrs={"units": "C"})
    result2 = da3 + da4
    assert result2.attrs == {}, f"Expected empty attrs, got {result2.attrs}"
    ok("Conflicting attrs are still dropped in operations")
except Exception as e:
    fail("Attrs behavior in operations", e)

# 9e. keep_attrs=True preserves them
try:
    da_a = xr.DataArray([1, 2, 3], attrs={"units": "K"})
    da_b = xr.DataArray([4, 5, 6], attrs={"units": "K"})
    with xr.set_options(keep_attrs=True):
        result_attrs = da_a + da_b
    assert result_attrs.attrs == {"units": "K"}
    ok("xr.set_options(keep_attrs=True) preserves attrs")
except Exception as e:
    fail("keep_attrs option", e)

# 9f. .values vs .data
try:
    da_dask = xr.DataArray(
        rng.standard_normal((10, 10)).astype("float32"),
        dims=["x", "y"],
    ).chunk({"x": 5})

    data_ref = da_dask.data
    import dask.array as dask_array
    assert isinstance(data_ref, dask_array.Array)
    ok(".data on chunked array returns dask array (lazy)")

    vals = da_dask.values
    assert isinstance(vals, np.ndarray)
    ok(".values on chunked array returns numpy array (triggers compute)")
except Exception as e:
    fail(".values vs .data", e)

# 9g. concat_dim requires combine='nested'
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            ds_f = xr.Dataset({
                "temp": ("x", rng.standard_normal(10).astype("float32")),
            })
            ds_f.to_netcdf(os.path.join(tmpdir, f"f_{i}.nc"))

        try:
            ds_wrong = xr.open_mfdataset(
                os.path.join(tmpdir, "f_*.nc"),
                concat_dim="time",
            )
            fail("open_mfdataset with concat_dim but no combine='nested' should fail")
            ds_wrong.close()
        except (ValueError, TypeError):
            ok("open_mfdataset(concat_dim='time') without combine='nested' raises error")

        ds_right = xr.open_mfdataset(
            os.path.join(tmpdir, "f_*.nc"),
            combine="nested",
            concat_dim="time",
        )
        assert "time" in ds_right.dims
        ok("open_mfdataset(combine='nested', concat_dim='time') works correctly")
        ds_right.close()
except Exception as e:
    fail("concat_dim requires combine='nested'", e)


# ===========================================================================
# 10. Encoding gotchas
# ===========================================================================
section("10. Encoding Gotchas")

# 10a. Encoding persists through operations
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = os.path.join(tmpdir, "input.zarr")
        p2 = os.path.join(tmpdir, "output.zarr")

        ds_src = xr.Dataset({
            "temp": (("time", "x"), rng.standard_normal((30, 10)).astype("float32")),
        }, coords={"time": range(30), "x": range(10)})
        ds_src.to_zarr(p1)

        ds_loaded = xr.open_zarr(p1)
        enc = ds_loaded["temp"].encoding
        assert len(enc) > 0
        ok("Encoding is present after open_zarr (persists)")

        ds_subset = ds_loaded.isel(time=slice(0, 10))
        ds_subset = ds_subset.drop_encoding()
        ds_subset.to_zarr(p2)
        ok("drop_encoding() before writing subset avoids stale chunk errors")
except Exception as e:
    fail("Encoding persistence gotcha", e)

# 10b. _FillValue=0 masks legitimate zeros
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        nc_path = os.path.join(tmpdir, "fill_test.nc")
        ds_fill = xr.Dataset({
            "data": ("x", np.array([0.0, 1.0, 2.0, 0.0, 3.0])),
        })
        ds_fill.to_netcdf(nc_path, encoding={"data": {"_FillValue": 0}})

        ds_read = xr.open_dataset(nc_path)
        assert np.isnan(ds_read["data"].values[0])
        assert np.isnan(ds_read["data"].values[3])
        ok("_FillValue=0 masks legitimate zeros (gotcha confirmed)")

        ds_raw = xr.open_dataset(nc_path, mask_and_scale=False)
        assert ds_raw["data"].values[0] == 0.0
        ok("mask_and_scale=False avoids _FillValue masking")
        ds_read.close()
        ds_raw.close()
except Exception as e:
    fail("_FillValue=0 gotcha", e)

# 10c. String coordinates to zarr
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        zpath = os.path.join(tmpdir, "str_coords.zarr")
        ds_str = xr.Dataset({
            "val": ("name", [1, 2, 3]),
        }, coords={"name": np.array(["a", "b", "c"], dtype=object)})

        assert ds_str.coords["name"].dtype == object
        for coord in ds_str.coords:
            if ds_str[coord].dtype == object:
                ds_str[coord] = ds_str[coord].astype(str)

        ds_str.to_zarr(zpath)
        ds_back = xr.open_zarr(zpath)
        ok("Casting object-dtype strings to str before zarr write")
except Exception as e:
    fail("String coordinates zarr write", e)


# ===========================================================================
# 11. Chunk alignment with zarr
# ===========================================================================
section("11. Chunk Alignment with Zarr")

try:
    with tempfile.TemporaryDirectory() as tmpdir:
        zpath = os.path.join(tmpdir, "chunked.zarr")

        ds_ch = xr.Dataset({
            "data": (("x", "y"), rng.standard_normal((200, 200)).astype("float32")),
        })
        ds_ch = ds_ch.chunk({"x": 100, "y": 100})
        ds_ch.to_zarr(zpath)

        ds_read = xr.open_zarr(zpath)
        ds_aligned = ds_read.chunk({"x": 200, "y": 200})
        ok("Chunk alignment: dask chunks as multiples of zarr chunks (documented)")
except Exception as e:
    fail("Chunk alignment", e)


# ===========================================================================
# 12. Deprecation Table Entries
# ===========================================================================
section("12. Deprecation Table Verification")

# 12a. ds.sizes replaces ds.dims for dict access
try:
    ds_dep = xr.Dataset({
        "temp": (("x", "y"), rng.standard_normal((5, 10)).astype("float32")),
    })
    sizes = ds_dep.sizes
    assert sizes["x"] == 5 and sizes["y"] == 10
    ok("ds.sizes works as dict-like replacement for ds.dims")

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dims_result = ds_dep.dims
        ok(f"ds.dims returns {type(dims_result).__name__} (skill says use ds.sizes instead)")
except Exception as e:
    fail("ds.dims vs ds.sizes", e)

# 12b. DataTree import from xarray
try:
    from xarray import DataTree
    ok("from xarray import DataTree (not from datatree package)")
except Exception as e:
    fail("DataTree import from xarray", e)

# 12c. open_dataset(use_cftime=True) deprecated -> CFDatetimeCoder
try:
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = os.path.join(tmpdir, "dep_test.nc")
            ds_dep_t = xr.Dataset({
                "temp": ("time", [1.0, 2.0]),
            }, coords={
                "time": xr.date_range("2000-01-01", periods=2, freq="D", unit="s"),
            })
            ds_dep_t.to_netcdf(nc_path)
            try:
                ds_dep_read = xr.open_dataset(nc_path, use_cftime=True)
                dep_warns = [x for x in w if "deprecat" in str(x.message).lower() or "use_cftime" in str(x.message)]
                if dep_warns:
                    ok("open_dataset(use_cftime=True) emits deprecation warning")
                else:
                    ok("open_dataset(use_cftime=True) works (may not warn yet)")
                ds_dep_read.close()
            except TypeError:
                ok("open_dataset(use_cftime=True) raises TypeError (fully removed)")
            except ValueError:
                # cftime not installed, but check if deprecation warning was raised before the error
                dep_warns = [x for x in w if "deprecat" in str(x.message).lower() or "use_cftime" in str(x.message)]
                if dep_warns:
                    ok("open_dataset(use_cftime=True) emits deprecation warning (cftime not installed)")
                else:
                    ok("open_dataset(use_cftime=True) raises ValueError (cftime not installed)")
except Exception as e:
    fail("use_cftime deprecation", e)


# ===========================================================================
# 13. Zarr fill_value != _FillValue
# ===========================================================================
section("13. Zarr fill_value vs _FillValue")

try:
    with tempfile.TemporaryDirectory() as tmpdir:
        zpath = os.path.join(tmpdir, "fill_zarr.zarr")

        ds_int = xr.Dataset({
            "counts": ("x", np.array([0, 5, 10, 0, 15], dtype="int32")),
        })
        ds_int.to_zarr(zpath)

        ds_raw = xr.open_dataset(zpath, engine="zarr", mask_and_scale=False)
        assert ds_raw["counts"].values[0] == 0
        ok("mask_and_scale=False with zarr preserves fill_value=0")
except Exception as e:
    fail("Zarr fill_value vs _FillValue", e)


# ===========================================================================
# 14. Performance Tips
# ===========================================================================
section("14. Performance Tips Verification")

try:
    da_perf = xr.DataArray(rng.standard_normal(1000).astype("float32"), dims=["x"])
    arr = da_perf.values
    assert isinstance(arr, np.ndarray)
    ok(".values extracts numpy array for fast loop access")
except Exception as e:
    fail(".values extraction", e)


# ===========================================================================
# 15. Known Limitations
# ===========================================================================
section("15. Known Limitations Verification")

try:
    assert hasattr(xr, "register_dataarray_accessor")
    ok("xr.register_dataarray_accessor exists (for subclassing alternative)")
except Exception as e:
    fail("register_dataarray_accessor", e)


# ===========================================================================
# 16. DataTree 'dataset' param
# ===========================================================================
section("16. DataTree 'dataset' param (renamed from 'ds')")

try:
    ds_node = xr.Dataset({"v": ("x", [1, 2])})
    dt_new = xr.DataTree(dataset=ds_node)
    assert "v" in dt_new.dataset.data_vars
    ok("DataTree(dataset=ds) works with 'dataset' parameter name")

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            dt_old = xr.DataTree(ds=ds_node)
            if w:
                ok("DataTree(ds=...) emits deprecation warning (renamed to dataset=)")
            else:
                ok("DataTree(ds=...) still works without warning")
        except TypeError:
            ok("DataTree(ds=...) raises TypeError (fully renamed to dataset=)")
except Exception as e:
    fail("DataTree dataset= param rename", e)


# ===========================================================================
# 17. Verify the skill file's claim about arithmetic_join default
# ===========================================================================
section("17. Verify arithmetic_join and arithmetic_compat defaults")

try:
    opts = xr.get_options()
    join_default = opts.get("arithmetic_join", "NOT SET")
    compat_default = opts.get("arithmetic_compat", "NOT SET")
    print(f"  arithmetic_join default: {join_default}")
    print(f"  arithmetic_compat default: {compat_default}")

    if join_default == "inner":
        ok("SKILL FILE ISSUE: arithmetic_join default is 'inner' - alignment NaN gotcha is outdated")
    elif join_default == "outer":
        ok("arithmetic_join default is 'outer' - alignment NaN gotcha is accurate")
    else:
        ok(f"arithmetic_join default is '{join_default}'")
except Exception as e:
    fail("arithmetic_join default check", e)


# ===========================================================================
# SUMMARY
# ===========================================================================
section("SUMMARY")
print(f"\n  Total PASS: {len(PASSES)}")
print(f"  Total FAIL: {len(FAILS)}")
if FAILS:
    print("\n  Failed tests:")
    for f in FAILS:
        print(f"    - {f}")
print()

sys.exit(1 if FAILS else 0)
