"""Handy helper functions used to compare/convert numpy to matlab arrays"""

import matlab
import numpy as np

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
