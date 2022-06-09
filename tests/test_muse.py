"""Unit tests for MUSE."""

import os
import unittest

from pathlib import Path

# Running these tests requires a matlab installation, as well as installing
# the matlab engine API for Python, which allows us to run matlab functions
# from within Python. For instructions on how to get this set up, see:
# https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
import matlab
import matlab.engine
import numpy as np

from muse.util import (
    argmax_grid, fft_base, make_xy_grid, mixing_matrix_from_n_mics,
    pad_at_high_freqs, r_est_from_clip_simplified, rsrp_from_dfted_clip_and_delays_fast,
    rsrp_from_xcorr_raw_and_delta_tau, rsrp_grid_from_clip_and_xy_grids, velocity_sound,
    xcorr_raw_from_dfted_clip
    )

import constants as c


MATLAB_MUSE_REPO = 'https://github.com/JaneliaSciComp/Muse.git'
DOWNLOAD_PATH = Path('tests/muse')
TOOLBOX_PATH = DOWNLOAD_PATH / 'toolbox'


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

def assert_np_matlab_almost_equal(np_arr, matlab_arr):
    """
    Helper function to compare a numpy array to a matlab array, to
    6 decimals of precision.
    """
    # first convert the numpy array to a matlab array
    np_as_matlab = np_to_matlab(np_arr)
    # if the numpy array is a vector,
    # conversion to matlab will automatically make it a row vector.
    # so if matlab output shape is a column vector,
    # we want to transpose np_arr to make the shapes match
    if len(np_arr.shape) == 1 and matlab_arr.size[1] == 1:
        # if so, transpose the numpy array before comparing
        np_as_matlab.reshape(np_as_matlab.size[1], np_as_matlab.size[0])
    
    # now compare the two
    np.testing.assert_array_almost_equal(
        np_as_matlab,
        matlab_arr
    )

def assert_np_matlab_bulk(list_np, list_matlab):
    """
    Helper function to compare multiple np and matlab arrays at once.

    Specifically, compare each item list_np[i] with list_matlab[i] using
    the function assert_np_matlab_almost_equal().
    
    Arguments:
        - list_np: list-like of numpy arrays
        - list_matlab: list-like of matlab arrays
    """
    for np_arr, matlab_arr in zip(list_np, list_matlab):
        assert_np_matlab_almost_equal(np_arr, matlab_arr)

class CompareWithMatlab(unittest.TestCase):
    """Compare the outputs of the python and matlab MUSE functions."""

    @classmethod
    def setUpClass(cls):
        """Temporarily download the MATLAB code before running tests."""
        # first clone the matlab repo locally
        os.system(f'git clone {MATLAB_MUSE_REPO} {DOWNLOAD_PATH.resolve()}')
        # start a matlab engine
        cls.eng = matlab.engine.start_matlab()
        # change to dir with the matlab muse files
        cls.eng.cd(str(TOOLBOX_PATH.resolve()))

    @classmethod
    def tearDownClass(cls):
        """Delete the MATLAB files to avoid cluttering git repo."""
        # delete the matlab git files
        os.system(f'rm -rf {DOWNLOAD_PATH.resolve()}')

    def test_velocity_sound(self):
        # test velocity of sound at a range of different temperature values
        for t in range(-20, 101, 10):
            # convert t to a float because if we pass an int,
            # matlab can't take square roots
            self.assertEqual(
                velocity_sound(t),
                self.eng.velocity_sound(float(t))
                )
    
    def test_pad_at_high_freqs(self):
        # pad_at_high_freqs is applied to a complex array of shape (N,)
        sample_input = np.arange(c.N, dtype=complex) - c.N/2
        # add imaginary components
        sample_input += np.random.random(c.N) * 1j
        # MUSE calls pad_at_high_freqs with second parameter N_line, where N_line = 8 * N
        MUSE_SR_SCALE_FACTOR = 8
        N_PADDED = MUSE_SR_SCALE_FACTOR * c.N
        # get python result
        py_result = pad_at_high_freqs(sample_input, N_PADDED)
        # get matlab result, converting N_PADDED to float because otherwise it
        # complains... i guess matlab engine just hates integers
        m_result = self.eng.pad_at_high_freqs(
            np_to_matlab(sample_input),
            float(N_PADDED)
            )
        # finally, compare the results
        assert_np_matlab_almost_equal(py_result, m_result)
    
    def test_mixing_matrix(self):
        for i in range(2, 10):
            # change py_result dtype to float to match matlab return type
            py_result = mixing_matrix_from_n_mics(i).astype(float)
            m_result = self.eng.mixing_matrix_from_n_mics(i)
            assert_np_matlab_almost_equal(py_result, m_result)
    
    def test_fft_base(self):
        # randomly pick sample_rate and N values
        for sample_rate, N in np.random.randint(10, 1000, size = (20, 2)):
            assert_np_matlab_almost_equal(
                fft_base(N, N / sample_rate),
                # convert N to float because otherwise matlab complains
                self.eng.fft_base(float(N), float(N / sample_rate))
                )
    
    def test_argmax_grid(self):
        # make arbitrary size x_grid, y_grid
        x_grid, y_grid = make_xy_grid(c.X_DIM, c.Y_DIM, resolution=c.GRID_RESOLUTION)
        # create a fake rsrp grid to go along with this,
        # scaled to values in the range [-50.0, 50.0)
        rsrp_grid = 100 * (np.random.random(size=x_grid.shape) - 0.5)
        # get results from python implementation
        py_results = argmax_grid(x_grid, y_grid, rsrp_grid)
        # get matlab results
        m_results = self.eng.argmax_grid(
            np_to_matlab(x_grid),
            np_to_matlab(y_grid),
            np_to_matlab(rsrp_grid),
            nargout=2
            )
        assert_np_matlab_bulk(py_results, m_results)
    
    def test_xcorr_raw(self):
        # xcorr_raw_from_dfted_clip takes three params:
        #   V: fft'd audio clip (N, n_mics)
        #   dt: duration between samples, or 1 / sample_rate
        #   M: mixing matrix (n_pairs, n_mics)

        # pick N_MICS = 4 and SAMPLE_RATE = 1250 arbitrarily, giving us
        # the dt and M values in constants.py

        # generate and analyze some random "audio" data
        for audio_input in np.random.random(size=(10, c.N, c.N_MICS)):
            V = np.fft.fft(audio_input, axis=0)
            py_result_tuple = xcorr_raw_from_dfted_clip(V, c.dt, c.M)
            # get matlab results
            # as always, convert int arrays (M) to float to appease the matlab engine
            m_result_tuple = self.eng.xcorr_raw_from_dfted_clip(
                np_to_matlab(V),
                c.dt,
                np_to_matlab(c.M.astype(float)),
                0, # verbosity
                nargout=2
            )
            # compare the tuples of arrays
            assert_np_matlab_bulk(py_result_tuple, m_result_tuple)

    def test_rsrp_from_xcorr_and_delta_tau(self):
        # arguments are:
        #  xcorr_raw_all: first output from xcorr_raw_from_dfted_clip, (8*N, n_pairs)
        #  tau_line:      second output from the same function (8*N, 1)
        #  tau_diff:      array storing the time of arrival difference for each pair of 
        #                 mics at each grid location
        #                 (n_pairs, n_r)

        # take a plausible tau_diff value from constants.py,
        # along with matching values of N, N_MICS, dt, and M

        # generate a bunch of random audio data
        for audio_input in np.random.random(size=(10, c.N, c.N_MICS)):
            # get our other two arguments
            V = np.fft.fft(audio_input, axis=0)
            xcorr_raw, tau_line = xcorr_raw_from_dfted_clip(V, c.dt, c.M)
            # get python results
            py_result_tuple = rsrp_from_xcorr_raw_and_delta_tau(
                xcorr_raw, tau_line, c.tau_diff
                )
            # get matlab results
            m_result_tuple = self.eng.rsrp_from_xcorr_raw_and_delta_tau(
                np_to_matlab(xcorr_raw),
                np_to_matlab(tau_line),
                np_to_matlab(c.tau_diff),
                nargout=2
                )
            assert_np_matlab_bulk(py_result_tuple, m_result_tuple)
    
    def test_rsrp_from_dfted_clip_and_delays_fast(self):
        # arguments are V, dt, tau, and a verbosity integer
        # just test on the sample data from constants.py,
        # changing up the audio input
        # generate a bunch of random audio data
        for audio_input in np.random.random(size=(10, c.N, c.N_MICS)):
            # get our other two arguments
            V = np.fft.fft(audio_input, axis=0)
            # get python result
            py_results = rsrp_from_dfted_clip_and_delays_fast(
                V,
                c.dt,
                c.tau,
                verbosity=0
            )
            # get matlab results
            m_results = self.eng.rsrp_from_dfted_clip_and_delays_fast(
                np_to_matlab(V),
                float(c.dt),
                np_to_matlab(c.tau),
                0, # verbosity
                nargout=3
            )
            # compare results
            assert_np_matlab_bulk(py_results, m_results)
    
    def test_rsrp_grid_from_clip_and_xy_grids(self):
        # this function has a ton of inputs, but most of them are taken
        # care of for us in constants.py
        # all we need to do is generate a bunch of fake audio data, as always
        for audio_input in np.random.random(size=(10, c.N, c.N_MICS)):
            py_result_tuple = rsrp_grid_from_clip_and_xy_grids(
                audio_input,
                c.SAMPLE_RATE,
                c.f_lo,
                c.f_hi,
                c.AIR_TEMP,
                c.x_grid,
                c.y_grid,
                c.MIC_POSITIONS,
                verbosity=0
            )
            m_result_tuple = self.eng.rsrp_grid_from_clip_and_xy_grids(
                np_to_matlab(audio_input),
                float(c.SAMPLE_RATE),
                float(c.f_lo),
                float(c.f_hi),
                float(c.AIR_TEMP),
                np_to_matlab(c.x_grid),
                np_to_matlab(c.y_grid),
                np_to_matlab(c.MIC_POSITIONS),
                0, # verbosity
                nargout=7
            )
            assert_np_matlab_bulk(py_result_tuple, m_result_tuple)
    
    def test_r_est_from_clip_simplified(self):
        # as in the previous test, tons of arguments but all taken care of
        # just generate audio
        for audio_input in np.random.random(size=(10, c.N, c.N_MICS)):
            py_result_tuple = r_est_from_clip_simplified(
                audio_input,
                c.SAMPLE_RATE,
                c.f_lo,
                c.f_hi,
                c.AIR_TEMP,
                c.x_grid,
                c.y_grid,
                in_cage=None, # unused argument
                R=c.MIC_POSITIONS,
                verbosity=0
            )
            m_result_tuple = self.eng.r_est_from_clip_simplified(
                np_to_matlab(audio_input),
                float(c.SAMPLE_RATE),
                float(c.f_lo),
                float(c.f_hi),
                float(c.AIR_TEMP),
                np_to_matlab(c.x_grid),
                np_to_matlab(c.y_grid),
                0, # in_cage, unused argument
                np_to_matlab(c.MIC_POSITIONS),
                0, # verbosity
                nargout=9
            )
            assert_np_matlab_bulk(py_result_tuple, m_result_tuple)
