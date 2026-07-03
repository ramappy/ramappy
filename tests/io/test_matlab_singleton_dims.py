import numpy as np
import scipy.io

from ramappy.io.matlab import read_matlab


def test_read_matlab_accepts_1x1xn_cube(tmp_path):
    # A 1x1 map stored as a 3-D cube (Y, X, S) should be accepted as SpectralMap.
    n_spectral = 17
    data = np.arange(n_spectral, dtype=np.float32).reshape(1, 1, n_spectral)
    x = np.linspace(400.0, 1800.0, n_spectral, dtype=np.float32)

    f = tmp_path / "tiny.mat"
    scipy.io.savemat(f, {"data": data, "x": x})

    hsi = read_matlab(
        str(f),
        data_label="data",
        x_label="x",
        as_hsi=True,
        # Common convention is (Y, X, S)
        axes_order=("Y", "X", "S"),
    )

    assert hsi.img_width == 1
    assert hsi.img_height == 1
    assert hsi.data.shape == (1, n_spectral)
    assert hsi.x.shape == (n_spectral,)


def test_read_matlab_accepts_vector_as_single_pixel_cube(tmp_path):
    # If the file stores data as a vector (N,) or (1,N)/(N,1), treat it as 1x1xN when as_hsi=True.
    n_spectral = 11
    data = np.arange(n_spectral, dtype=np.float32)
    x = np.arange(n_spectral, dtype=np.float32)

    f = tmp_path / "tiny_vector.mat"
    scipy.io.savemat(f, {"data": data, "x": x})

    hsi = read_matlab(
        str(f),
        data_label="data",
        x_label="x",
        as_hsi=True,
        axes_order=("X", "Y", "S"),
    )

    assert hsi.img_width == 1
    assert hsi.img_height == 1
    assert hsi.data.shape == (1, n_spectral)
