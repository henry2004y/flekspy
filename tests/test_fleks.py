import pytest
import os
import itertools
import numpy as np
import xarray as xr
import polars as pl

from flekspy.tp import interpolate_at_times
import flekspy as fs
from flekspy.util import download_testfile

import matplotlib

matplotlib.use("agg")


@pytest.fixture(scope="module")
def idl_data_files(setup_test_data):
    """Fixture to provide paths to the IDL test data files."""
    filenames = (
        "1d__raw_2_t25.60000_n00000258.out",
        "z=0_fluid_region0_0_t00001640_n00010142.out",
        "3d_raw.out",
        "bx0_mhd_6_t00000100_n00000352.out",
    )
    return [os.path.join(setup_test_data, file) for file in filenames]


@pytest.fixture(scope="module")
def amrex_data_files(setup_test_data):
    """Fixture to provide paths to the AMReX test data files."""
    files = ("z=0_fluid_region0_0_t00001640_n00010142.out", "3d*amrex")
    return [os.path.join(setup_test_data, file) for file in files]


@pytest.fixture(scope="module")
def particle_data_dirs(setup_test_data):
    """Fixture to provide paths to the particle test data directories."""
    return (
        os.path.join(setup_test_data, "test_particles"),
        os.path.join(setup_test_data, "test_particles_PBEG"),
    )


class TestIDL:
    def test_load(self, idl_data_files):
        ds = fs.load(idl_data_files[0])
        assert isinstance(ds, xr.Dataset)
        assert ds.attrs["time"] == 25.6
        assert ds.coords["x"][1] == -126.5
        assert np.isclose(ds["Bx"][2].item(), 0.22360679775)

        ds = fs.load(idl_data_files[3])
        assert ds.Rho[1].item() == 6.8823977459

    def test_load_error(self):
        with pytest.raises(FileNotFoundError):
            ds = fs.load("None")

    def test_extract(self, idl_data_files):
        ds = fs.load(idl_data_files[1])
        d = ds.interp(x=-28000.0, y=0.0)
        assert len(d) == 28

    def test_slice(self, idl_data_files):
        ds = fs.load(idl_data_files[2])
        slice_data = ds.idl.get_slice("z", 0.0)
        assert isinstance(slice_data, xr.Dataset)
        assert len(slice_data) == 14
        assert slice_data.sizes["x"] == 8
        assert slice_data.sizes["y"] == 8
        assert slice_data["absdivB"].shape == (8, 8)

    def test_plot(self, idl_data_files):
        ds = fs.load(idl_data_files[0])
        ds.p.plot()
        ds = fs.load(idl_data_files[1])
        ds.rhoS0.plot.pcolormesh(x="x", y="y")
        ds["Bx"].plot.pcolormesh(x="x", y="y")
        ds.plot.streamplot(x="x", y="y", u="Bx", v="By", color="w")
        ds = fs.load(idl_data_files[3])
        ds.Rho.ugrid.plot.contourf()
        assert True


class TestAMReX:
    def test_load(self, amrex_data_files):
        ds = fs.load(amrex_data_files[0])
        assert ds["uxS0"][2, 1] == np.float32(-131.71918)
        assert ds["uxS1"].shape == (601, 2)

    def test_pic(self, amrex_data_files):
        ds = fs.load(amrex_data_files[1], use_yt_loader=True)
        assert ds.domain_left_edge[0].v == -0.016
        dc = ds.get_slice("z", 0.5)
        assert dc.data["particle_id"][0].value == 216050.0
        assert dc.__repr__().startswith("variables")

    def test_phase(self, amrex_data_files):
        ds = fs.load(amrex_data_files[1], use_yt_loader=True)
        x_field = "p_uy"
        y_field = "p_uz"
        z_field = "p_w"
        xleft = [-0.016, -0.01, ds.domain_left_edge[2]]
        xright = [0.016, 0.01, ds.domain_right_edge[2]]

        ## Select and plot the particles inside a box defined by xleft and xright
        region = ds.box(xleft, xright)
        x, y, w = ds.get_phase(
            x_field,
            y_field,
            z_field,
            region=region,
            domain_size=(xleft[0], xright[0], xleft[1], xright[1]),
        )
        assert x.shape == (128,) and w.max() == 2.8024863240162035e23
        pp = ds.plot_phase(
            x_field,
            y_field,
            z_field,
            region=region,
            unit_type="si",
            x_bins=100,
            y_bins=32,
            domain_size=(xleft[0], xright[0], xleft[1], xright[1]),
        )
        ## Plot inside a sphere
        center = [0, 0, 0]
        radius = 1
        # Object sphere is defined in yt/data_objects/selection_objects/spheroids.py
        sp = ds.sphere(center, radius)
        pp = ds.plot_particles(
            "p_x", "p_y", "p_w", region=sp, unit_type="planet", x_bins=32, y_bins=32
        )
        pp = ds.plot_phase(
            "p_uy", "p_uz", "p_w", region=sp, unit_type="si", x_bins=16, y_bins=16
        )
        f = fs.extract_phase(pp)
        assert f[0].size == 16 and f[2].shape == (16, 16)

    def test_amrex_particle_loader_default(self, setup_test_data):
        ds = fs.load(os.path.join(setup_test_data, "3d_particle*amrex"))
        assert isinstance(ds, fs.amrex.AMReXParticleData)


@pytest.fixture(scope="class")
def particle_tracker(particle_data_dirs):
    """Fixture to initialize FLEKSTP for planetary units."""
    return fs.FLEKSTP(particle_data_dirs[0], iSpecies=1, unit="planetary")


@pytest.fixture(scope="class")
def particle_tracker_si(particle_data_dirs):
    """Fixture to initialize FLEKSTP for SI units and caching."""
    return fs.FLEKSTP(particle_data_dirs[1], iSpecies=1, use_cache=True, unit="SI")


class TestParticles:
    def test_particles(self, particle_tracker):
        pIDs = particle_tracker.getIDs()
        assert particle_tracker.__repr__().startswith("Particles")
        assert pIDs[0] == (0, 5121)
        pt = particle_tracker[10].collect()
        assert pt["x"][0] == -0.031386006623506546
        assert pt["vx"][3] == 5.870406312169507e-05
        assert pt["vy"][5] == 4.103916944586672e-05
        assert pt["vz"].shape == (8,)
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            pt.select("unknown")
        x = particle_tracker.read_initial_condition(pIDs[10])
        assert x[1] == pt["x"][0]
        x = particle_tracker.read_final_condition(particle_tracker.IDs[10])
        assert x[1] == pt["x"][-1]
        ids, pData = particle_tracker.read_particles_at_time(0.0, doSave=False)
        assert ids[1][1] == 5129
        ax = particle_tracker.plot_location(pData[0:2, :])
        assert ax["A"].get_xlim()[1] == -0.03136133626103401
        with pytest.raises(Exception):
            particle_tracker.read_particles_at_time(-10.0, doSave=False)
        with pytest.raises(Exception):
            particle_tracker.read_particles_at_time(10.0, doSave=False)

    def test_particle_select(self, particle_tracker):
        from flekspy.tp import Indices

        def f_select(tp, pid):
            pData = tp.read_initial_condition(pid)
            inRegion = pData[Indices.X] > 0 and pData[Indices.Y] > 0
            return inRegion

        pSelected = particle_tracker.select_particles(f_select)
        assert len(pSelected) == 2560

    def test_trajectory(self, particle_tracker):
        pIDs = particle_tracker.getIDs()
        ax = particle_tracker.plot_trajectory(pIDs[0], type="single")
        assert ax.get_xlim()[1] == 2.140599811077118
        ax = particle_tracker.plot_trajectory(
            pIDs[0], type="single", xaxis="y", yaxis="z", ax=ax
        )
        assert ax.get_xlim()[0] == -0.12292625373229385
        ax = particle_tracker.plot_trajectory(pIDs[0], type="xv")
        assert ax[1].get_xlim()[1] == 2.140599811077118
        ax = particle_tracker.plot_trajectory(pIDs[0])
        assert ax[1][0].get_xlim()[1] == 2.140599811077118

    def test_particle_cache(self, particle_data_dirs):
        tp = fs.FLEKSTP(
            particle_data_dirs[0], iSpecies=1, use_cache=True, unit="planetary"
        )
        pID = tp.getIDs()[0]

        # First access, should be read from file
        trajectory1 = tp[pID]

        # Second access, should be from cache
        trajectory2 = tp[pID]

        # Check if they are the same object
        assert trajectory1 is trajectory2

    def test_read_particle_trajectory_key_error(self, particle_tracker):
        with pytest.raises(KeyError):
            particle_tracker.read_particle_trajectory((-1, -1))

    def test_read_particle_trajectory_value_error(self, particle_tracker, monkeypatch):
        pID = particle_tracker.IDs[0]
        # Ensure the cache is clean for this test
        if pID in particle_tracker._trajectory_cache:
            del particle_tracker._trajectory_cache[pID]

        monkeypatch.setattr(
            particle_tracker,
            "_get_particle_raw_data",
            lambda pID: np.array([], dtype=np.float32),
        )
        with pytest.raises(ValueError):
            particle_tracker.read_particle_trajectory(pID)

    def test_invalid_unit(self, particle_data_dirs):
        with pytest.raises(ValueError):
            fs.FLEKSTP(particle_data_dirs[0], iSpecies=1, unit="invalid_unit")

    def test_interpolate_at_times_float32(self):
        df = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "x": [0, 10, 20, 30, 40],
                "y": [40, 30, 20, 10, 0],
                "z": [0, 5, 10, 5, 0],
            }
        ).with_columns(pl.col("time").cast(pl.Float32))

        times_to_interpolate = [0.5, 1.5, 2.5]

        interpolated_df = interpolate_at_times(df, times_to_interpolate)

        assert interpolated_df.shape == (3, 4)
        assert interpolated_df["time"].dtype == pl.Float32
        assert np.all(
            np.isclose(interpolated_df["time"].to_list(), times_to_interpolate)
        )
        assert np.all(np.isclose(interpolated_df["x"].to_list(), [5.0, 15.0, 25.0]))
        assert np.all(np.isclose(interpolated_df["y"].to_list(), [35.0, 25.0, 15.0]))
        assert np.all(np.isclose(interpolated_df["z"].to_list(), [2.5, 7.5, 7.5]))

    def test_EBG(self, particle_tracker_si):
        from flekspy.tp import plot_integrated_energy

        p0_collected = particle_tracker_si[0].collect()
        assert p0_collected.item(0, 7) == 224199.65625  # bx
        assert p0_collected.item(0, 16) == 2194893.75  # dbydx
        pid = particle_tracker_si.getIDs()[0]
        assert particle_tracker_si.get_pitch_angle(pid)[0] == np.float32(57.661438)
        vx, vy, vz = (
            p0_collected.item(0, 4),
            p0_collected.item(0, 5),
            p0_collected.item(0, 6),
        )
        ke = particle_tracker_si.get_kinetic_energy(vx, vy, vz)
        assert np.isclose(ke, 3.361357097373841e-17)
        pt_lazy = particle_tracker_si[pid]
        assert np.isclose(
            particle_tracker_si.get_ExB_drift(pt_lazy).item(0, 1),
            3.9656504668528214e-05,
        )
        # kappa z, not y
        assert np.isclose(
            particle_tracker_si._calculate_curvature(pt_lazy).collect().item(0, -1),
            -0.4797530472278595,
        )
        assert np.isclose(
            particle_tracker_si.get_curvature_drift(pt_lazy).item(0, 0),
            -4.17402271497159e-23,
        )
        assert np.isclose(
            particle_tracker_si.get_gradient_drift(pt_lazy).item(0, 1),
            -6.209680085413732e-26,
        )
        assert np.isclose(
            particle_tracker_si.get_polarization_drift(pt_lazy).item(0, 2),
            1.1582171530455036e-19,
        )

        df_drifts = particle_tracker_si.integrate_drift_accelerations(pid)
        assert "Wp_integrated" in df_drifts.columns
        plot_integrated_energy(df_drifts)
        particle_tracker_si.analyze_drifts(pid)
        particle_tracker_si.analyze_drift(pid, "ExB")

        rc2rl = particle_tracker_si.get_adiabaticity_parameter(pt_lazy)[0]
        assert np.isclose(rc2rl, 3.2436217368142828e16)

        tcross = particle_tracker_si.find_shock_crossing_time(
            particle_tracker_si.getIDs()[0], b_threshold_factor=1
        )
        assert tcross == 0.0
        s_up_dn = particle_tracker_si.get_shock_up_down_states(
            particle_tracker_si.getIDs(),
            delta_t_up=0.1,
            delta_t_down=0.1,
            b_threshold_factor=1,
            verbose=False,
        )
        assert s_up_dn[1]["time"][1] == 1.2570836544036865

    def test_energy_change_guiding_center(self, particle_tracker_si):
        pid = particle_tracker_si.getIDs()[0]

        df = particle_tracker_si.get_energy_change_guiding_center(pid)

        # Check if the output is a Polars DataFrame
        assert isinstance(df, pl.DataFrame)

        # Check for expected columns
        expected_columns = {
            "time",
            "dW_parallel",
            "dW_betatron",
            "dW_fermi",
            "dW_total",
        }
        assert expected_columns.issubset(df.columns)

        # Check that dW_total is the sum of the components
        total_sum = df["dW_parallel"] + df["dW_betatron"] + df["dW_fermi"]
        assert np.allclose(df["dW_total"].to_numpy(), total_sum.to_numpy())

        # Check that the number of rows is correct
        pt_len = len(particle_tracker_si[pid].collect())
        assert len(df) == pt_len
        # TODO Check numerical values!

    def test_analyze_drifts_energy_change(self, particle_tracker_si, tmp_path):
        pid = particle_tracker_si.getIDs()[0]
        outname = tmp_path / "test_energy_change.png"
        particle_tracker_si.analyze_drifts_energy_change(pid, outname=str(outname))
        assert outname.exists()

    def test_work_energy_verification(self, particle_tracker_si, tmp_path):
        pid = particle_tracker_si.getIDs()[0]
        outname = tmp_path / "test_work_energy.png"
        particle_tracker_si.plot_work_energy_verification(pid, outname=str(outname))
        assert outname.exists()

    def test_save_trajectories_h5(self, particle_tracker, tmp_path):
        pIDs = particle_tracker.getIDs()[:2]
        filename = tmp_path / "trajectories.h5"

        particle_tracker.save_trajectories(pIDs, filename=str(filename))

        import h5py
        assert filename.exists()
        with h5py.File(filename, "r") as f:
            expected_keys = [f"ID_{pid[0]}_{pid[1]}" for pid in pIDs]
            assert all(key in f.keys() for key in expected_keys)

            # Verify data and attributes for the first particle
            dset = f[expected_keys[0]]
            assert dset.shape[0] > 0

            original_columns = particle_tracker[pIDs[0]].collect().columns
            saved_columns = dset.attrs["columns"]
            assert all(original_columns == saved_columns)

    def test_save_trajectories_h5_int_list(self, particle_tracker, tmp_path):
        pIDs = [0, 1]
        filename = tmp_path / "trajectories_int.h5"

        particle_tracker.save_trajectories(pIDs, filename=str(filename))

        import h5py
        assert filename.exists()
        with h5py.File(filename, "r") as f:
            # Need to get the actual pID tuples from the particle_tracker
            expected_pIDs = [particle_tracker.IDs[i] for i in pIDs]
            expected_keys = [f"ID_{pid[0]}_{pid[1]}" for pid in expected_pIDs]
            assert all(key in f.keys() for key in expected_keys)

            # Verify data and attributes for the first particle
            dset = f[expected_keys[0]]
            assert dset.shape[0] > 0

            original_columns = particle_tracker[expected_pIDs[0]].collect().columns
            saved_columns = dset.attrs["columns"]
            assert all(original_columns == saved_columns)

    def test_save_trajectories_h5_numpy_array(self, particle_tracker, tmp_path):
        pIDs = np.array([0, 1])
        filename = tmp_path / "trajectories_np.h5"

        particle_tracker.save_trajectories(pIDs, filename=str(filename))

        import h5py
        assert filename.exists()
        with h5py.File(filename, "r") as f:
            # Need to get the actual pID tuples from the particle_tracker
            expected_pIDs = [particle_tracker.IDs[i] for i in pIDs]
            expected_keys = [f"ID_{pid[0]}_{pid[1]}" for pid in expected_pIDs]
            assert all(key in f.keys() for key in expected_keys)

            # Verify data and attributes for the first particle
            dset = f[expected_keys[0]]
            assert dset.shape[0] > 0

            original_columns = particle_tracker[expected_pIDs[0]].collect().columns
            saved_columns = dset.attrs["columns"]
            assert all(original_columns == saved_columns)


def load_and_benchmark(files):
    """
    Helper function for benchmarking flekspy loading.
    """
    fs.load(files[0])
    return fs.load(files[1])


def test_load_idl(benchmark, idl_data_files):
    result = benchmark(load_and_benchmark, idl_data_files)
    assert isinstance(result, xr.Dataset)


def load_test_particle_trajectories(tp, pIDs):
    """
    Load a subset of particle trajectories for benchmarking.
    """
    for pID in itertools.islice(pIDs, 100):
        tp.read_particle_trajectory(pID)


def test_load_tp(benchmark, particle_tracker):
    """
    Benchmark loading particle trajectories.
    """
    pIDs = particle_tracker.getIDs()
    benchmark(load_test_particle_trajectories, particle_tracker, pIDs)


def get_drifts(tp, pid):
    """Helper function to calculate particle drifts."""
    pt_lazy = tp[pid]
    tp.get_curvature_drift(pt_lazy)
    tp.get_gradient_drift(pt_lazy)
    tp.get_ExB_drift(pt_lazy)


def test_drift_tp(benchmark, particle_tracker_si):
    """
    Benchmark particle drift calculations.
    """
    pid = particle_tracker_si.getIDs()[0]
    benchmark(get_drifts, particle_tracker_si, pid)
