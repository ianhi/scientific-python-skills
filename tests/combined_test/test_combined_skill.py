"""
Combined test of zarr, xarray, and icechunk skill files.

This script exercises a realistic scientific data management workflow
combining all three libraries, using ONLY patterns documented in the
skill files as guidance.
"""

import numpy as np
import xarray as xr
import zarr
import icechunk
from icechunk.xarray import to_icechunk


def main():
    print("=" * 70)
    print("COMBINED SKILL FILE TEST: zarr + xarray + icechunk")
    print("=" * 70)

    # ----------------------------------------------------------------
    # Step 1: Create an icechunk repository (in-memory storage)
    # Pattern from icechunk skill: icechunk.in_memory_storage() + Repository.create()
    # ----------------------------------------------------------------
    print("\n[Step 1] Creating icechunk repository with in-memory storage...")
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.create(storage)
    print(f"  Repository created. Branches: {repo.list_branches()}")

    # ----------------------------------------------------------------
    # Step 2: Generate realistic xarray data
    # Pattern from xarray skill: xr.Dataset with dims, coords, attrs
    # Pattern from xarray skill: xr.date_range with unit="s" for datetime coords
    # ----------------------------------------------------------------
    print("\n[Step 2] Generating realistic xarray climate dataset...")
    np.random.seed(42)

    # xarray skill: use unit="s" to control time resolution
    times = xr.date_range("2000-01-01", periods=100, freq="D", unit="s")
    lats = np.linspace(-90, 90, 45)
    lons = np.linspace(-180, 180, 90)

    # xarray skill: Dataset constructor with (dims, data) tuples
    ds = xr.Dataset(
        {
            "temperature": (
                ("time", "lat", "lon"),
                np.random.randn(100, 45, 90).astype(np.float32) * 10 + 280,
            ),
            "precipitation": (
                ("time", "lat", "lon"),
                np.abs(np.random.randn(100, 45, 90).astype(np.float32)) * 5,
            ),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
        attrs={
            "title": "Synthetic Climate Dataset",
            "source": "Combined skill file test",
            "conventions": "CF-1.8",
        },
    )
    print(f"  Dataset created: {ds}")
    print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"  Variables: {list(ds.data_vars)}")

    # ----------------------------------------------------------------
    # Step 3: Write to icechunk using to_icechunk (NOT to_zarr)
    # Pattern from icechunk skill: to_icechunk(ds, session, mode="w")
    # Pattern from icechunk skill: session.commit("message")
    # Critical: use to_icechunk, not ds.to_zarr(session.store, ...)
    # ----------------------------------------------------------------
    print("\n[Step 3] Writing dataset to icechunk with to_icechunk...")
    session = repo.writable_session("main")
    to_icechunk(ds, session, mode="w")
    snapshot_v1 = session.commit("Initial dataset v1.0")
    print(f"  Committed snapshot: {snapshot_v1}")

    # ----------------------------------------------------------------
    # Step 4: Tag the release as "v1.0"
    # Pattern from icechunk skill: repo.create_tag("name", snapshot_id=id)
    # ----------------------------------------------------------------
    print("\n[Step 4] Tagging release as 'v1.0'...")
    repo.create_tag("v1.0", snapshot_id=snapshot_v1)
    print(f"  Tags: {repo.list_tags()}")

    # ----------------------------------------------------------------
    # Step 5: Create a "dev" branch and append 50 more timesteps
    # Pattern from icechunk skill: repo.lookup_branch() + repo.create_branch()
    # Pattern from icechunk skill: new session after commit
    # Pattern from icechunk skill: to_icechunk(ds_new, session, append_dim="time")
    # ----------------------------------------------------------------
    print("\n[Step 5] Creating 'dev' branch and appending data...")
    main_snap = repo.lookup_branch("main")
    repo.create_branch("dev", snapshot_id=main_snap)
    print(f"  Branches: {repo.list_branches()}")

    # Generate 50 more timesteps
    new_times = xr.date_range("2000-04-10", periods=50, freq="D", unit="s")
    ds_new = xr.Dataset(
        {
            "temperature": (
                ("time", "lat", "lon"),
                np.random.randn(50, 45, 90).astype(np.float32) * 10 + 282,
            ),
            "precipitation": (
                ("time", "lat", "lon"),
                np.abs(np.random.randn(50, 45, 90).astype(np.float32)) * 5,
            ),
        },
        coords={
            "time": new_times,
            "lat": lats,
            "lon": lons,
        },
        attrs=ds.attrs,
    )

    session = repo.writable_session("dev")
    to_icechunk(ds_new, session, append_dim="time")
    snapshot_dev = session.commit("Append 50 timesteps on dev branch")
    print(f"  Dev branch committed: {snapshot_dev}")

    # ----------------------------------------------------------------
    # Step 6: Read from the v1.0 tag and verify 100 timesteps (time-travel)
    # Pattern from icechunk skill: repo.readonly_session(tag="v1.0")
    # Pattern from icechunk skill: xr.open_zarr(session.store, consolidated=False)
    # ----------------------------------------------------------------
    print("\n[Step 6] Time-travel read from v1.0 tag...")
    session_v1 = repo.readonly_session(tag="v1.0")
    ds_v1 = xr.open_zarr(session_v1.store, consolidated=False)
    time_count_v1 = len(ds_v1.time)
    print(f"  v1.0 time steps: {time_count_v1}")
    assert time_count_v1 == 100, f"Expected 100 timesteps at v1.0, got {time_count_v1}"
    print("  PASS: v1.0 has exactly 100 timesteps")

    # ----------------------------------------------------------------
    # Step 7: Read from the dev branch and verify 150 timesteps
    # Pattern from icechunk skill: repo.readonly_session(branch="dev")
    # ----------------------------------------------------------------
    print("\n[Step 7] Reading from dev branch...")
    session_dev = repo.readonly_session(branch="dev")
    ds_dev = xr.open_zarr(session_dev.store, consolidated=False)
    time_count_dev = len(ds_dev.time)
    print(f"  Dev branch time steps: {time_count_dev}")
    assert time_count_dev == 150, f"Expected 150 timesteps on dev, got {time_count_dev}"
    print("  PASS: dev branch has exactly 150 timesteps")

    # ----------------------------------------------------------------
    # Step 8: Process data with apply_ufunc
    # Pattern from xarray skill: apply_ufunc with dask='parallelized',
    #   input_core_dims, output_core_dims, output_dtypes
    # Pattern from xarray skill: core dims moved to LAST axis (-1)
    # ----------------------------------------------------------------
    print("\n[Step 8] Computing running standardized anomalies with apply_ufunc...")

    # Chunk the dev data for dask processing
    # xarray skill: ds.chunk() for dask-backed arrays
    ds_dev_chunked = ds_dev.chunk({"time": -1, "lat": 15, "lon": 30})

    def running_standardized_anomaly(arr):
        """Compute running standardized anomaly along last axis (time).

        Core dims are moved to the LAST axis by apply_ufunc.
        """
        # Cumulative mean and std along time axis (last axis, -1)
        cumsum = np.cumsum(arr, axis=-1)
        n = np.arange(1, arr.shape[-1] + 1)
        cumulative_mean = cumsum / n

        # Cumulative variance (using Welford-like approach)
        cumsum_sq = np.cumsum(arr**2, axis=-1)
        cumulative_var = cumsum_sq / n - cumulative_mean**2
        cumulative_std = np.sqrt(np.maximum(cumulative_var, 1e-10))

        # Standardized anomaly
        anomaly = (arr - cumulative_mean) / cumulative_std
        return anomaly

    # xarray skill: apply_ufunc pattern 2 - reduction/transform along core dims
    # MUST specify output_dtypes with dask='parallelized'
    anomalies = xr.apply_ufunc(
        running_standardized_anomaly,
        ds_dev_chunked["temperature"],
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        dask="parallelized",
        output_dtypes=[float],
    )
    print(f"  Anomaly result shape: {anomalies.sizes}")
    print(f"  Anomaly is dask-backed: {hasattr(anomalies.data, 'dask')}")

    # Compute the anomalies to get actual values
    # xarray skill: .compute() to trigger lazy computation
    anomalies_computed = anomalies.compute()
    print(f"  Anomaly computed. Mean: {float(anomalies_computed.mean()):.4f}")
    print("  PASS: apply_ufunc with dask='parallelized' succeeded")

    # ----------------------------------------------------------------
    # Step 9: Write processed data to a new group in icechunk
    # Pattern from icechunk skill (gotcha): multiple groups need mode="a"
    #   for to_zarr, but we use to_icechunk
    # Pattern from icechunk skill: new writable session for each commit
    # ----------------------------------------------------------------
    print("\n[Step 9] Writing processed anomalies to 'processed/anomalies' group...")

    # Create a dataset from the anomalies for writing
    ds_anomalies = xr.Dataset(
        {"temperature_anomaly": anomalies_computed},
        attrs={
            "title": "Running Standardized Temperature Anomalies",
            "source_branch": "dev",
            "processing": "cumulative standardized anomaly",
        },
    )

    # icechunk skill: new writable session (previous one is read-only after commit)
    session = repo.writable_session("dev")

    # icechunk skill (gotcha #7): multiple groups -- first group already written,
    # subsequent groups need mode="a" when using to_zarr.
    # For to_icechunk, we write to a group path.
    # The icechunk skill shows to_zarr with group= for multiple groups.
    # Let's try to_zarr directly since this is non-dask data (icechunk skill says
    # "to_zarr with non-dask data" is fine).
    ds_anomalies.to_zarr(
        session.store,
        group="processed/anomalies",
        zarr_format=3,
        consolidated=False,
        mode="a",  # icechunk skill: mode="a" for additional groups
    )
    snapshot_processed = session.commit("Add processed standardized anomalies")
    print(f"  Committed processed data: {snapshot_processed}")

    # Verify we can read it back
    session_read = repo.readonly_session(branch="dev")
    ds_check = xr.open_zarr(
        session_read.store,
        group="processed/anomalies",
        consolidated=False,
    )
    print(f"  Read back processed data: {list(ds_check.data_vars)}")
    assert "temperature_anomaly" in ds_check.data_vars
    print("  PASS: Processed data written and read from group successfully")

    # ----------------------------------------------------------------
    # Step 10: Verify zarr v3 format
    # Pattern from zarr skill: zarr.open_group, group members, array info
    # ----------------------------------------------------------------
    print("\n[Step 10] Verifying zarr v3 format...")

    # Open the root group using zarr directly
    session_verify = repo.readonly_session(branch="dev")
    root = zarr.open_group(session_verify.store, mode="r")

    # zarr skill: list(grp.members()) to iterate
    print("  Root group members:")
    for name, obj in root.members():
        print(f"    {name}: {type(obj).__name__}")

    # Check zarr_format on an array
    # zarr skill: group['path/to/array'] for access
    temp_arr = root["temperature"]
    print(f"  temperature array zarr_format: {temp_arr.metadata.zarr_format}")
    assert temp_arr.metadata.zarr_format == 3, (
        f"Expected zarr_format=3, got {temp_arr.metadata.zarr_format}"
    )
    print("  PASS: zarr v3 format confirmed")

    # Check the processed group too
    processed_group = root["processed/anomalies"]
    print(f"  processed/anomalies group members:")
    for name, obj in processed_group.members():
        print(f"    {name}: {type(obj).__name__}")
        if isinstance(obj, zarr.Array):
            print(f"      zarr_format: {obj.metadata.zarr_format}")
            assert obj.metadata.zarr_format == 3

    # ----------------------------------------------------------------
    # Summary: Browse history
    # Pattern from icechunk skill: repo.ancestry(branch=...)
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMMIT HISTORY (dev branch):")
    print("=" * 70)
    for ancestor in repo.ancestry(branch="dev"):
        print(f"  {ancestor.id[:12]}... : {ancestor.message}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)

    # Print version info for debugging
    print(f"\nVersions: zarr={zarr.__version__}, xarray={xr.__version__}, icechunk={icechunk.__version__}")


if __name__ == "__main__":
    main()
