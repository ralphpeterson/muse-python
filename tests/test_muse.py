"""Unit tests for MUSE."""

import os
import unittest

from pathlib import Path

import matlab
import numpy as np

from muse.util import velocity_sound


MATLAB_MUSE_REPO = "https://github.com/JaneliaSciComp/Muse.git"
DOWNLOAD_PATH = Path('tests/muse')
TOOLBOX_PATH = DOWNLOAD_PATH / 'toolbox'

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
        os.system(f'rm -rf {DOWNLOAD_PATH.resolve()}')

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


    def test_velocity_sound(self):
        for t in range(-20, 101, 10):
            # convert t to a float because if we pass an int,
            # matlab can't take square roots
            self.assertEqual(
                velocity_sound(t),
                self.eng.velocity_sound(float(t))
                )
