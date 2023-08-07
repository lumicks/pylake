import os
import timeit
import tempfile
import contextlib

import numpy as np

import lumicks.pylake as lk


def _generate_kymo_for_tracking(duration, line_count, samples_per_pixel=1):
    """Generate a kymograph that can be used for tracking lines

    Parameters
    ----------
    duration : float
        Duration of the kymograph
    line_count : int
        Number of trackable lines
    samples_per_pixel : int
        Channel samples per pixel
    """
    from lumicks.pylake.tests.data.mock_confocal import generate_kymo

    pos = np.arange(0, duration, 0.01)
    x = np.arange(-2, 2, 0.1)
    kymo_data = (100 * np.exp(-5 * (x[:, np.newaxis] - np.sin(pos)) ** 2) + 10).astype(int)
    kymo_data = np.tile(kymo_data, [line_count, 1])
    kymograph = generate_kymo(
        "bench_kymo",
        kymo_data,
        pixel_size_nm=100,
        start=np.int64(20e9),
        dt=np.int64(1e9),
        samples_per_pixel=samples_per_pixel,
        line_padding=10,
    )
    return kymograph


def _generate_blank_kymo_data(samples=1000000):
    """This is a different function since generating a junk data kymo is significantly faster than
    generating one with sensible image content."""
    from lumicks.pylake.tests.data.mock_confocal import MockConfocalFile

    counts = np.ones(samples)
    infowave = np.ones(samples)
    infowave[::2] = 0
    infowave[::10] = 2

    return MockConfocalFile.from_streams(
        start=0,
        dt=int(1e9),
        axes=[0],
        num_pixels=[100],
        pixel_sizes_nm=[1000],
        infowave=infowave,
        red_photon_counts=counts,
        blue_photon_counts=counts,
        green_photon_counts=counts,
    )


def _generate_test_stack(tiff_fn, pixel_size=0.1):
    """Generate a tiff stack readable by ImageStack that can be used for benchmarking"""
    from lumicks.pylake.tests.data.mock_widefield import write_tiff_file, make_irm_description

    x, y = np.meshgrid(np.arange(-5, 5, pixel_size), np.arange(-5, 5, pixel_size))
    img_data = np.stack([np.sinc(a * (x**2 + y**2)) for a in np.arange(0.1, 2.0, 0.1)])
    write_tiff_file(img_data, make_irm_description(1, 16), n_frames=100, filename=tiff_fn)


class _Benchmark:
    name: str
    loops: int

    def __call__(self):
        return contextlib.contextmanager(self.context)()

    def context(self):
        raise NotImplementedError


class _KymoImage(_Benchmark):
    name = "Reconstructing kymo images"
    loops = 60

    def context(self):
        confocal_file, metadata, stop = _generate_blank_kymo_data()
        yield lambda: lk.kymo.Kymo("big_kymo", confocal_file, 0, stop, metadata).get_image("red")


class _KymoTimestamps(_Benchmark):
    name = "Reconstructing kymo timestamps"
    loops = 45

    def context(self):
        confocal_file, metadata, stop = _generate_blank_kymo_data()
        yield lambda: lk.kymo.Kymo("big_kymo", confocal_file, 0, stop, metadata).timestamps


class _Tracking(_Benchmark):
    name = "Tracking lines on a kymograph"
    loops = 1

    def context(self):
        kymo = _generate_kymo_for_tracking(12, 11)
        yield lambda: lk.track_greedy(kymo, "red", track_width=1, pixel_threshold=20)


class _Refinement(_Benchmark):
    name = "Gaussian line refinement"
    loops = 1

    def context(self):
        kymo_tracking = _generate_kymo_for_tracking(1, 4)
        lines = lk.track_greedy(kymo_tracking, "red", track_width=1, pixel_threshold=20)
        yield lambda: lk.refine_tracks_gaussian(
            lines, 10000, refine_missing_frames=True, overlap_strategy="multiple"
        )


class _FdFit(_Benchmark):
    loops = 1

    def __init__(self, interpolated=False):
        self.interpolated = interpolated  # generally faster at inverting the model
        self.name = f"Fd fitting ({'interpolation' if interpolated else 'rootfinding'})"

    def context(self):
        # The interpolated method is a lot faster, so we can do more points in that case
        force = np.arange(0.1, 20.0, 0.95 * (0.07 if self.interpolated else 1))
        true = {"m/Lp": 50.0, "m/Lc": 1024 * 0.34, "m/St": 1200.0, "kT": 4.11}
        distance = lk.ewlc_odijk_distance("m")(force, true) + 0.02

        def bench():
            # Deliberately use the manually inverted form (slower)
            manually_inverted = (lk.ewlc_odijk_distance("m") + lk.distance_offset("m")).invert(
                independent_min=0.0, independent_max=30, interpolate=self.interpolated
            )

            fit = lk.FdFit(manually_inverted)
            fit.add_data("dataset", force, distance)
            fit["kT"].value = 4.11
            fit["m/Lc"].value = 1000 * 0.34
            fit["m/d_offset"].value = 0.02
            fit["m/d_offset"].fixed = True
            fit.fit()

        yield bench


class _ReadTIFF(_Benchmark):
    name = "Reading TIFF files (disk)"
    loops = 2

    def context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "bench_tiff.tiff")
            _generate_test_stack(filename)

            def read_image_stack():
                stack = lk.ImageStack(filename)
                stack.get_image()
                stack._src.close()  # Explicitly close, otherwise cleanup will fail

            yield read_image_stack


class _ProcessTIFF(_Benchmark):
    name = "Processing TIFF files"
    loops = 2

    def context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tiff_fn = os.path.join(tmpdir, "bench_tiff.tiff")
            _generate_test_stack(tiff_fn, pixel_size=0.23)
            stack = lk.ImageStack(tiff_fn)
            yield lambda: stack.define_tether((10, 10), (50, 50)).get_image()
            stack._src.close()  # Explicitly close, otherwise cleanup will fail


def benchmark(repeat=5):
    benchmarks = [
        _KymoImage(),
        _KymoTimestamps(),
        _Tracking(),
        _Refinement(),
        _FdFit(interpolated=False),
        _FdFit(interpolated=True),
        _ReadTIFF(),
        _ProcessTIFF(),
    ]
    str_format = f".<{np.max([len(b.name) for b in benchmarks]) + 3}"

    print(f"pylake v{lk.__version__} (only compare results between matching versions)")
    print("Benchmarks:")
    total = 0.0
    for bench in benchmarks:
        print(f"- {bench.name:{str_format}} ", end="", flush=True)
        with bench() as func:
            times = timeit.repeat(func, number=bench.loops, repeat=repeat)
        dt = np.median(times)
        total += dt
        print(f"{dt:.2f} +- {np.std(times):.2f} seconds")
    print(f"Total: {total:.2f} seconds")


if __name__ == "__main__":
    benchmark()
