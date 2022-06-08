"""Unit tests for MUSE."""

import os
import unittest

from pathlib import Path

import matlab
import matlab.engine
import numpy as np

from muse.util import (
    argmax_grid, fft_base, make_xy_grid, mixing_matrix_from_n_mics,
    pad_at_high_freqs, velocity_sound
    )


MATLAB_MUSE_REPO = "https://github.com/JaneliaSciComp/Muse.git"
DOWNLOAD_PATH = Path('tests/muse')
TOOLBOX_PATH = DOWNLOAD_PATH / 'toolbox'

# Declare floats almost equal if the first five decimal places agree
ALMOST_EQUAL_PRECISION = 5

def np_to_matlab(arr: np.ndarray):
        """Convert a numpy array to a matlab array of matching type."""
        # first convert arr to a nested list, which matlab arr constructor expects
        nested = arr.tolist()
        # return the appropriately typed matlab array
        if arr.dtype == np.int_:
            return matlab.int64(nested)
        if arr.dtype == np.bool_:
            return matlab.logical(nested)
        if arr.dtype == np.float_:
            return matlab.double(nested)
        if arr.dtype == np.complex_:
            return matlab.double(nested, is_complex=True)


class CompareWithMatlab(unittest.TestCase):
    """Compare the outputs of the python and matlab MUSE functions."""

    @classmethod
    def setUpClass(cls):
        # first clone the matlab repo locally
        os.system(f'git clone {MATLAB_MUSE_REPO} {DOWNLOAD_PATH.resolve()}')
        # start a matlab engine
        cls.eng = matlab.engine.start_matlab()
        # change to dir with the matlab muse files
        cls.eng.cd(str(TOOLBOX_PATH.resolve()))

    @classmethod
    def tearDownClass(cls):
        # delete the matlab git files
        os.system(f'rm -rf {DOWNLOAD_PATH.resolve()}')

    def test_velocity_sound(self):
        for t in range(-20, 101, 10):
            # convert t to a float because if we pass an int,
            # matlab can't take square roots
            self.assertEqual(
                velocity_sound(t),
                self.eng.velocity_sound(float(t))
                )
    
    def test_pad_at_high_freqs(self):
        # this function is applied to a complex array of shape (N,)
        sample_input = np.arange(-50, 50, dtype=complex)
        # make it complex
        sample_input += np.random.random(100) * 1j
        # MUSE calls this function with second parameter N_line, where N_line = 8 * N
        MUSE_SR_SCALE_FACTOR = 8
        N_PADDED = MUSE_SR_SCALE_FACTOR * len(sample_input)
        # get python result
        py_result = pad_at_high_freqs(sample_input, N_PADDED)
        # convert input and python result to matlab
        py_result = np_to_matlab(py_result)
        m_input = np_to_matlab(sample_input)
        # get matlab result, converting N_PADDED to float because otherwise it
        # complains... i guess matlab engine just hates integers
        m_result = self.eng.pad_at_high_freqs(m_input, float(N_PADDED))
        # finally, transpose py_result. converting to a matlab array adds
        # an extra dimension, making the result a row vector,
        # when the matlab code for pad_at_high_freqs returns a column vector
        py_result.reshape(py_result.size[1], py_result.size[0])
        np.testing.assert_array_almost_equal(py_result, m_result, decimal=ALMOST_EQUAL_PRECISION)
    
    def test_mixing_matrix(self):
        for i in range(2, 10):
            # change py_result dtype to float to match matlab return type
            py_result = mixing_matrix_from_n_mics(i).astype(float)
            m_result = self.eng.mixing_matrix_from_n_mics(i)
            self.assertEqual(np_to_matlab(py_result), m_result)
    
    def test_fft_base(self):
        # randomly pick sample_rate and N values
        for sample_rate, N in np.random.randint(10, 1000, size = (20, 2)):
            self.assertEqual(
                np_to_matlab(fft_base(N, N / sample_rate)),
                # convert to float because otherwise matlab complains
                self.eng.fft_base(float(N), float(N / sample_rate))
                )
    
    def test_argmax_grid(self):
        # make arbitrary size x_grid, y_grid
        X_LEN, Y_LEN = np.random.randint(1, 100, size=2)
        RESOLUTION = np.random.random()
        x_grid, y_grid = make_xy_grid(X_LEN, Y_LEN, resolution=RESOLUTION)
        # create a fake rsrp grid to go along with this,
        # scaled to values in the range [-50.0, 50.0)
        rsrp_grid = 100 * (np.random.random(size=x_grid.shape) - 0.5)
        # get results from python implementation
        py_rsrp_grid, py_rsrp_max = argmax_grid(x_grid, y_grid, rsrp_grid)
        # convert the grids to matlab
        m_x_grid = np_to_matlab(x_grid)
        m_y_grid = np_to_matlab(y_grid)
        m_rsrp_grid = np_to_matlab(rsrp_grid)
        # get matlab result
        m_rsrp_grid, m_rsrp_max = self.eng.argmax_grid(m_x_grid, m_y_grid, m_rsrp_grid, nargout=2)
        np.testing.assert_array_almost_equal(np_to_matlab(py_rsrp_grid), m_rsrp_grid, decimal = ALMOST_EQUAL_PRECISION)
        self.assertAlmostEqual(py_rsrp_max, m_rsrp_max)

