"""Clean, Pythonic interface for main MUSE functions."""

import itertools
from typing import Tuple
import numpy as np
from function_files.util import argmax_grid, r_est_from_clip_simplified

def make_xy_grid(x_len: float, y_len: float, resolution=0.00025) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the dimensions of an arena and a desired grid resolution,
    return a tuple (x_grid, y_grid), where x_grid is a 2D numpy array
    storing the x-coordinate of every gridpoint. Similarly, y_grid stores
    the y-coordinate of every gridpoint.

    Args:
        x_len: Length of the x dimension of the arena, in meters.
        y_len: Length of the y dimension, in meters.
        resolution: Desired spatial resolution of grid, in meters.

    Returns:
        A tuple (x_grid, y_grid) as above.
    """
    x_dim = int(x_len/resolution)
    y_dim = int(y_len/resolution)

    x_coords = np.linspace(0, x_len, x_dim)
    y_coords = np.linspace(0, y_len, y_dim)

    return np.meshgrid(x_coords, y_coords)


def r_est_naive(
    v: np.ndarray,
    fs: int,
    f_lo: int,
    f_hi: int,
    temp: float,
    x_len: float,
    y_len: float,
    resolution: float,
    mic_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Naively estimate sound source location from microphone array data.

    Specifically, calculate the Reduced Steered Response Power (RSRP)
    at a grid of points on the room floor, spaced by the provided resolution.
    Then, return the point with the max RSRP value as well as the grid itself.

    Args:
        v: Array of microphone signals. Expected shape: (n_mics, n_samples)
        fs: The sampling frequency of audio data, in Hz.
        f_lo: The lower bound of frequency band used, in Hz. Frequencies
            below this are zeroed after applying the FFT.
        f_hi: The upper bound of frequency band used, in Hz.
        temp: Air temperature at which data was collected, in degrees C.
        x_len: The length of the room, in meters.
        y_len: The width of the room, in meters.
        resolution: Desired spatial resolution with which to estimate location.
        mic_positions: Array storing the position of each microphone in
            Cartesian coordinates. Expected shape: (n_mics, 3)

    Returns:
        A tuple (r_est, rsrp_grid).
        
        r_est: Array of shape (2,1) storing the x and y coordinates of the
            estimated sound source location.
        rsrp_grid: Array storing the calculated RSRP values at points on
            the arena floor, spaced apart by the specified resolution.
    """
    # make grids using the provided room dimensions
    x_grid, y_grid = make_xy_grid(x_len, y_len, resolution=resolution)

    # get the rsrp values

    in_cage = None # unused param
    # note: transpose v and mic_positions because matlab expects
    # v to be shape (n_samples, n_mics) and mic_positions to be shape
    # (3, n_mics)
    r_est, _, rsrp_grid, _, _, _, _, _, _ = r_est_from_clip_simplified(
        v.T, fs, f_lo, f_hi, temp, x_grid, y_grid, in_cage, mic_positions.T, verbosity=0
        )

    return r_est, rsrp_grid


def r_est_jackknife(
    v: np.ndarray,
    fs: int,
    f_lo: int,
    f_hi: int,
    temp: float,
    x_len: float,
    y_len: float,
    resolution: float,
    mic_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate sound source location from microphone array data using the
    Jackknife method, from Warren 2018 (https://pubmed.ncbi.nlm.nih.gov/29309793).

    Essentially, this function systematically excludes one microphone and
    naively estimates the sound source location using the remaining microphones.
    Repeating this for each microphone yields n_mics point estimates, which are
    then averaged and returned.

    Args:
        v: Array of microphone signals. Expected shape: (n_mics, n_samples)
        fs: The sampling frequency of audio data, in Hz.
        f_lo: The lower bound of frequency band used, in Hz. Frequencies
            below this are zeroed after applying the FFT.
        f_hi: The upper bound of frequency band used, in Hz.
        temp: Air temperature at which data was collected, in degrees C.
        x_len: The length of the room, in meters.
        y_len: The width of the room, in meters.
        resolution: Desired spatial resolution with which to estimate location.
        mic_positions: Array storing the position of each microphone in
            Cartesian coordinates. Expected shape: (n_mics, 3)
    
    Returns:
        A tuple (avg_est, r_estimates, rsrp_grids).

        avg_est: Array of shape (2,1) storing the x and y coordinates of the
            estimated sound source location, averaged across all point estimates.
        r_estimates: List of arrays, where the ith array represents the sound
            source location estimate when microphone i was removed.
        rsrp_grids: List of arrays, where the ith array stores the
            RSRP grid result when microphone i was removed. Each grid is an
            array of calculated RSRP values at points on the arena floor,
            spaced apart by the specified resolution.
    """
    # to speed up the jackknife procedure, rather than applying r_est_naive
    # to each subset of data with one mic removed, we directly calculate
    # the RSRP values and estimates from the cross-correlation values

    # why would we want to do this? because RSRP is defined to be the
    # sum of these cross-correlations. so this way, we avoid repeatedly
    # calling r_est_naive

    # so, run r_est_from_clip_simplified and just consider the xcorrs

    # make grids using the provided room dimensions
    x_grid, y_grid = make_xy_grid(x_len, y_len, resolution=resolution)

    in_cage = None # unused param
    # note: transpose v and mic_positions because matlab expects
    # v to be shape (n_samples, n_mics) and mic_positions to be shape
    # (3, n_mics)
    _, _, _, _, _, _, _, _, xcorr_per_pair_grid = r_est_from_clip_simplified(
        v.T, fs, f_lo, f_hi, temp, x_grid, y_grid,
        in_cage, mic_positions.T, verbosity=0
        )
    
    # now that we have the cross-correlations for each pair of microphones,
    # we can selectively sum them, removing one microphone at a time

    # xcorr_per_pair_grid has shape (xgrid.shape[0], xgrid.shape[1], n_pairs),
    # where the pairs are ordered as follows for a four mic scenario:
    # (1, 2)  pair with mic 1 and mic 2
    # (1, 3)
    # (1, 4)
    # (2, 3)
    # (2, 4)
    # (3, 4)

    # so to get the RSRP values for every mic save i, add up every row
    # whose corresponding mic pair doesn't include mic i

    def idxs_to_include(n_mics, mic_to_remove):
        """
        Get the indices of the pairs that should still be included after
        removing microphone `mic_to_remove`.
        """
        # get all pairs of microphones (i, j) with i < j
        pairs = itertools.combinations(range(n_mics), 2)
        # keep the indices of pairs that don't contain mic `mic_to_remove`
        idxs = [i for i, pair in enumerate(pairs) if mic_to_remove not in pair]
        return idxs

    r_estimates = []
    rsrp_grids = []

    N_MICS = v.shape[0]

    for i in range(N_MICS):
        # find the indices along the pairs axis that don't include mic i
        idxs = idxs_to_include(N_MICS, i)
        # calculate the RSRP values by summing up the cross correlations
        # for each pair
        rsrp_grid = xcorr_per_pair_grid[:, :, idxs].sum(axis=2)
        rsrp_grids.append(rsrp_grid)
        r_est, _ = argmax_grid(x_grid, y_grid, rsrp_grid)
        r_estimates.append(r_est)
    
    avg_est = np.mean(r_estimates, axis=0)

    return avg_est, r_estimates, rsrp_grids