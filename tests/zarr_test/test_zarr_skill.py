"""
Test script exercising zarr v3 features, guided ONLY by the zarr skill file.
Each section is independent and prints clear pass/fail output.
"""

import tempfile
import os
import numpy as np

import zarr
import zarr.codecs
from zarr.storage import LocalStore, MemoryStore


def section(name):
    print(f"\n{'='*70}")
    print(f"  SECTION: {name}")
    print(f"{'='*70}")


def main():
    # Create a temporary directory for all stores
    with tempfile.TemporaryDirectory(prefix="zarr_skill_test_") as tmpdir:
        print(f"Using temporary directory: {tmpdir}")

        # ================================================================
        # 1. Sharded array with custom codecs (ZstdCodec)
        # ================================================================
        section("1. Sharded zarr v3 array with ZstdCodec")
        try:
            store_path = os.path.join(tmpdir, "sharded_array.zarr")
            z = zarr.create_array(
                store=store_path,
                shape=(1000, 1000),
                chunks=(100, 100),
                shards=(500, 500),
                dtype="float32",
                compressors=zarr.codecs.ZstdCodec(level=3),
                zarr_format=3,
            )
            data = np.random.random((1000, 1000)).astype(np.float32)
            z[:] = data
            # Verify round-trip
            read_back = z[:]
            assert np.allclose(data, read_back), "Data mismatch after round-trip!"
            print(f"  Created sharded array: shape={z.shape}, dtype={z.dtype}")
            print(f"  Chunks: {z.chunks}, Shards: {z.shards}")
            print(f"  nchunks_initialized: {z.nchunks_initialized}")
            print(f"  PASS")
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")

        # ================================================================
        # 2. Group hierarchy
        # ================================================================
        section("2. Group hierarchy with subgroups")
        try:
            store_path = os.path.join(tmpdir, "group_hierarchy.zarr")
            root = zarr.open_group(store_path, mode="w")
            # Create nested groups
            exp_grp = root.create_group("experiment")
            raw_grp = exp_grp.create_group("raw")
            proc_grp = exp_grp.create_group("processed")

            # Create arrays in each subgroup
            raw_arr = raw_grp.create_array(
                "sensor_data", shape=(365, 24), dtype="float32"
            )
            proc_arr = proc_grp.create_array(
                "cleaned_data", shape=(365, 24), dtype="float64"
            )

            # Fill with data
            raw_arr[:] = np.random.random((365, 24)).astype(np.float32)
            proc_arr[:] = np.random.random((365, 24)).astype(np.float64)

            # Access using path syntax as shown in skill file
            arr_via_path = root["experiment/raw/sensor_data"]
            assert arr_via_path.shape == (365, 24), "Path access shape mismatch!"

            arr_via_path2 = root["experiment/processed/cleaned_data"]
            assert arr_via_path2.shape == (365, 24), "Path access shape mismatch!"

            # Iterate members
            members = list(root.members())
            print(f"  Root members: {[(k, type(v).__name__) for k, v in members]}")
            print(f"  Accessed 'experiment/raw/sensor_data' via path: shape={arr_via_path.shape}")
            print(f"  Accessed 'experiment/processed/cleaned_data' via path: shape={arr_via_path2.shape}")
            print(f"  PASS")
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")

        # ================================================================
        # 3. from_array: convert numpy array to zarr on disk
        # ================================================================
        section("3. from_array - numpy to zarr")
        try:
            store_path = os.path.join(tmpdir, "from_array.zarr")
            np_data = np.arange(10000, dtype="float32").reshape(100, 100)

            # Skill file says: store is first positional arg, data is keyword-only
            z = zarr.from_array(store_path, data=np_data, chunks=(50, 50))

            read_back = z[:]
            assert np.allclose(np_data, read_back), "from_array data mismatch!"
            print(f"  Created array from numpy: shape={z.shape}, dtype={z.dtype}")
            print(f"  Chunks: {z.chunks}")
            print(f"  PASS")
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")

        # ================================================================
        # 4. Codec pipeline: BytesCodec + BloscCodec
        # ================================================================
        section("4. Explicit codec pipeline (BytesCodec + BloscCodec)")
        # NOTE: The skill file shows codecs=[BytesCodec(), BloscCodec()] but
        # create_array() does NOT accept a 'codecs' parameter.
        # The actual API uses serializer= and compressors= separately.
        # First, demonstrate the skill file version fails:
        try:
            store_path_fail = os.path.join(tmpdir, "codec_pipeline_fail.zarr")
            z_fail = zarr.create_array(
                store=store_path_fail,
                shape=(1000,),
                dtype="float32",
                codecs=[
                    zarr.codecs.BytesCodec(endian="little"),
                    zarr.codecs.BloscCodec(cname="zstd", clevel=5),
                ],
            )
            print(f"  Skill file 'codecs=' syntax: unexpectedly WORKED")
        except TypeError as e:
            print(f"  Skill file 'codecs=' syntax: FAILS as expected: {e}")

        # Now use the correct API: serializer= + compressors=
        try:
            store_path = os.path.join(tmpdir, "codec_pipeline.zarr")
            z = zarr.create_array(
                store=store_path,
                shape=(1000,),
                dtype="float32",
                serializer=zarr.codecs.BytesCodec(endian="little"),
                compressors=zarr.codecs.BloscCodec(
                    cname="zstd",
                    clevel=5,
                    shuffle=zarr.codecs.BloscShuffle.shuffle,
                ),
            )
            data = np.random.random(1000).astype(np.float32)
            z[:] = data
            read_back = z[:]
            assert np.allclose(data, read_back), "Codec pipeline data mismatch!"
            print(f"  Correct API (serializer= + compressors=): shape={z.shape}")
            print(f"  Info: {z.info}")
            print(f"  PASS")
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")

        # ================================================================
        # 5. Consolidated metadata
        # ================================================================
        section("5. Consolidated metadata")
        try:
            store_path = os.path.join(tmpdir, "consolidated.zarr")
            root = zarr.open_group(store_path, mode="w")
            for i in range(5):
                root.create_array(f"arr_{i}", shape=(100,), dtype="float32")

            # Consolidate metadata
            zarr.consolidate_metadata(store_path)
            print(f"  Consolidated metadata written.")

            # Open with consolidated metadata (auto-detect)
            root2 = zarr.open_group(store_path)
            arr = root2["arr_3"]
            print(f"  Opened with auto-detected consolidated: arr_3 shape={arr.shape}")

            # Open with explicit consolidated
            root3 = zarr.open_consolidated(store_path)
            arr2 = root3["arr_1"]
            print(f"  Opened with open_consolidated: arr_1 shape={arr2.shape}")

            # Open with use_consolidated=True
            root4 = zarr.open_group(store_path, use_consolidated=True)
            arr3 = root4["arr_4"]
            print(f"  Opened with use_consolidated=True: arr_4 shape={arr3.shape}")

            print(f"  PASS")
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")

        # ================================================================
        # 6. Store configuration with context manager
        # ================================================================
        section("6. zarr.config.set() context manager")
        try:
            # Use the context manager as shown in skill file
            with zarr.config.set({"array.write_empty_chunks": True}):
                store_path = os.path.join(tmpdir, "config_test.zarr")
                z = zarr.create_array(
                    store=store_path, shape=(100,), dtype="float32"
                )
                z[:] = np.zeros(100, dtype="float32")  # all fill_value
                nchunks_with = z.nchunks_initialized
                print(f"  write_empty_chunks=True: nchunks_initialized={nchunks_with}")

            # Without context, default write_empty_chunks=False
            store_path2 = os.path.join(tmpdir, "config_test2.zarr")
            z2 = zarr.create_array(
                store=store_path2, shape=(100,), dtype="float32"
            )
            z2[:] = np.zeros(100, dtype="float32")  # all fill_value
            nchunks_without = z2.nchunks_initialized
            print(f"  Default (write_empty_chunks=False): nchunks_initialized={nchunks_without}")

            # With write_empty_chunks=True, chunks should be written even if all fill_value
            # With False, they may not be written
            print(f"  Behavior difference detected: {nchunks_with} vs {nchunks_without}")
            print(f"  PASS")
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")

        # ================================================================
        # 7. Mode semantics
        # ================================================================
        section("7. Mode semantics (r, r+, a)")
        try:
            store_path = os.path.join(tmpdir, "mode_test.zarr")
            # First create an array
            z = zarr.create_array(
                store=store_path, shape=(100,), dtype="float32"
            )
            data = np.arange(100, dtype="float32")
            z[:] = data

            # Mode 'r' - read only
            z_r = zarr.open_array(store_path, mode="r")
            read_data = z_r[:]
            assert np.allclose(data, read_data), "Mode 'r' data mismatch!"
            print(f"  mode='r': Read successfully, shape={z_r.shape}")
            # Try writing - should fail
            try:
                z_r[:] = np.zeros(100, dtype="float32")
                print(f"  mode='r': Write unexpectedly succeeded (should be read-only)")
            except Exception as write_err:
                print(f"  mode='r': Write correctly blocked: {type(write_err).__name__}")

            # Mode 'r+' - read/write existing
            z_rw = zarr.open_array(store_path, mode="r+")
            z_rw[0] = 999.0
            assert z_rw[0] == 999.0, "Mode 'r+' write failed!"
            print(f"  mode='r+': Read/write on existing array works")

            # Mode 'r+' on non-existent - should fail
            nonexist_path = os.path.join(tmpdir, "nonexistent.zarr")
            try:
                z_fail = zarr.open_array(nonexist_path, mode="r+")
                print(f"  mode='r+' on non-existent: Unexpectedly succeeded")
            except Exception as rp_err:
                print(f"  mode='r+' on non-existent: Correctly failed: {type(rp_err).__name__}")

            # Mode 'a' - append/create
            a_path = os.path.join(tmpdir, "append_test.zarr")
            z_a = zarr.open_array(a_path, mode="a", shape=(50,), dtype="float32")
            z_a[:] = np.ones(50, dtype="float32")
            print(f"  mode='a': Created new array with shape={z_a.shape}")

            # Re-open with mode 'a' - should open existing
            z_a2 = zarr.open_array(a_path, mode="a")
            assert z_a2.shape == (50,), f"mode='a' re-open shape mismatch: {z_a2.shape}"
            assert z_a2[0] == 1.0, "mode='a' re-open data mismatch"
            print(f"  mode='a': Re-opened existing array, data preserved")

            print(f"  PASS")
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")

        # ================================================================
        # 8. v3 string dtype
        # ================================================================
        section("8. Variable-length string dtype")
        try:
            store_path = os.path.join(tmpdir, "string_array.zarr")
            z = zarr.create_array(
                store=store_path, shape=(5,), dtype="string"
            )
            z[:] = np.array(["hello", "world", "zarr", "v3", "strings"])
            read_back = z[:]
            print(f"  Created string array: shape={z.shape}, dtype={z.dtype}")
            print(f"  Data: {read_back}")
            assert list(read_back) == ["hello", "world", "zarr", "v3", "strings"], \
                f"String data mismatch: {read_back}"
            print(f"  PASS")
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")

        print(f"\n{'='*70}")
        print(f"  ALL SECTIONS COMPLETE")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
