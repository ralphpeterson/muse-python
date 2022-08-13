"""Direct port of MUSE from the original at https://github.com/JaneliaSciComp/Muse."""

from typing import Any, Tuple

import numpy as np

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


def velocity_sound(T: float) -> float:
    """
    Calculate the velocity of sound (m/s) based on the air temperature (degrees C).

    The below formula is taken from Signals, Sound and Sensation (Hartmann, 1998).
    For more information, see:
    https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/velocity_sound.m
    """
    return 331.3*np.sqrt(1+(T/273.16))


def pad_at_high_freqs(x: np.ndarray, n_padded: int) -> np.ndarray:
    """
    Pad an input in the frequency domain to increase sampling rate once an IFFT
    is applied and we return to the time domain.

    Given the result of a FFT, x, and a desired output length, n_padded, return
    a new array of the desired length padded with zeroes.

    See comments on https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/pad_at_high_freqs.m
    """

    num_elements = len(x)
    ratio = n_padded / num_elements

    # split the array into negative and nonnegative sections
    num_nonnegative = int(np.ceil(num_elements / 2))
    num_negative = num_elements - num_nonnegative
    x_nonneg_freq = x[:num_nonnegative]
    x_neg_freq = x[num_nonnegative:]

    # then pad the middle with zeros
    x_padded = np.zeros((n_padded,), dtype=x.dtype)
    # note: we multiply the amplitude by `ratio` to avoid modifying the
    # original amplitudes in the time domain
    x_padded[:num_nonnegative] = ratio * x_nonneg_freq
    x_padded[-num_negative:] = ratio * x_neg_freq
    return x_padded


def rsrp_from_xcorr_raw_and_delta_tau(
    xcorr_raw_all: np.ndarray,
    tau_line: np.ndarray,
    tau_diff: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the RSRP values at each grid point from a pre-computed array of
    cross-correlation values, a list of values at which xcorr was computed,
    and the true tau values at the gridpoints.

    For more information, see:
    https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/rsrp_from_xcorr_raw_and_delta_tau.m

    Args:
        xcorr_raw_all: Array storing the pre-computed cross-correlation values
            for each microphone pair (i, j), where i < j, at various
            time values. Expected shape: (N * 8, n_pairs)
      - tau_line: Array storing the time values at which the cross-correlation
            was precomputed. Expected shape: (N * 8, 1)
      - tau_diff: True time delay differences for each microphone pair.
            Expected shape: (n_pairs, n_r)

    Returns:
        A tuple (rsrp, rsrp_per_pairs). Note: the rsrp values are just the sum
        over all pairs of the cross-correlation values.

        rsrp: Array of length n_r storing the RSRP values at each gridpoint.
        xcorr_per_pairs: Array of shape (n_gridpts, n_pairs) storing xcorr values
            at each grid point calculated from each pair.
    """
    # get interval at which we calculated cross-correlations
    tau_0 = tau_line[0, 0]
    dtau = (tau_line[-1, -1] - tau_0) / (len(tau_line) - 1)

    # compute the cross-correlation for each tau_diff value by
    # interpolating linearly into the values we precomputed

    # find values in tau_line that sandwich each tau_diff
    k_real = ((tau_diff - tau_0) / dtau).T
    k_lo = np.floor(k_real).astype(int)
    k_hi = k_lo + 1
    # the "x coordinate" of the linear interpolation,
    # storing how far away the true t value is from the lower
    # value at which we pre-computed xcorr
    w_hi = k_real - k_lo

    # now perform the linear interpolation, calculating the xcorr
    # values at each true tau_diff value
    _, j = np.indices(k_lo.shape)
    xcorr_per_pairs = (1 - w_hi) * xcorr_raw_all[k_lo, j] + w_hi * xcorr_raw_all[k_hi, j]
    rsrp = xcorr_per_pairs.sum(axis=1)
    return rsrp, xcorr_per_pairs


def mixing_matrix_from_n_mics(n_mics: int) -> np.ndarray:
    """
    Return an array of shape (n_pairs, n_mics) that helps calculate the pairwise
    differences in time delay values for each microphone.
    
    Specifically, create an array M of shape (n_pairs, n_mics), where
        n_pairs = (n_mics) * (n_mics - 1) / 2,
    with the following property:
        Multiplying M with an array V of shape (n_mics, k) returns a new 
        array whose rows correspond to the pairwise differences V[i, :] - V[j, :]
        for each microphone pair (i, j) with i < j.
    
    For example: mixing_matrix_from_n_mics(3) should return:
        np.ndarray([
            [1, -1, 0],
            [1, 0, -1],
            [0, 1, -1]
        ])
    
    And mixing_matrix_from_n_mics(4) should return:
        np.ndarray([
            [1, -1, 0, 0],
            [1, 0, -1, 0],
            [1, 0, 0, -1],
            [0, 1, -1, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1]
        ])

    For more detail, see:
    https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/mixing_matrix_from_n_mics.m

    Args:
        n_mics: The number of mics in the microphone array.
    Returns:
        A matrix M as described above.
    """
    n_pairs = n_mics * (n_mics - 1) // 2
    # helper for the user in case their microphone data V is of the wrong shape
    if n_mics > 10:
        print(f'mixing_matrix_from_n_mics: There appear to be {n_mics} microphones. Perhaps V should be transposed?')
    M = np.zeros((n_pairs, n_mics), dtype=int)
    i = 0
    j = 1
    # for each i, create a row in M where the ith entry is 1 and the
    # jth entry is -1, for all i < j < n_mics. All other entries in M
    # should be zero.
    for row in range(n_pairs):
        M[row, i] = 1
        M[row, j] = -1
        j += 1
        if j >= n_mics:
            i += 1
            j = i + 1
    return M


def fft_base(N: int, dx: float) -> np.ndarray:
    """
    Using the number of samples and the bin width in the frequency domain,
    return an array storing the frequencies associated with the fft results.
    
    For more detail, see:
    https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/fft_base.m

    Args:
        N: the number of samples
        dx: the frequency domain bin width, calculated as 1 / (N * sample_rate)
    
    Returns:
        An array of shape (N,) where each entry represents a frequency at which
        the fft result was calculated.
    """
    hi_x_sample_index = np.ceil(N/2).astype('int')
    # generate the range of positive frequency values
    x_pos = dx*np.linspace(0,hi_x_sample_index-1,hi_x_sample_index)
    # and the negatives
    x_neg = dx*np.linspace(-(N-hi_x_sample_index), -1, N-hi_x_sample_index)

    return np.concatenate([x_pos, x_neg])[:, np.newaxis]


def rsrp_from_dfted_clip_and_delays_fast(
    V: np.ndarray,
    dt: float,
    tau: np.ndarray,
    verbosity: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate RSRP values at each gridpoint from the FFT'd audio input,
    timestep between samples, and array storing the expected time delay tau
    between each gridpoint and each microphone.

    For more, see:
    https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/rsrp_from_dfted_clip_and_delays_fast.m

    Args:
        V: FFT'd result of mic data. Expected shape: (N, n_mics), where N is
            the number of samples.
        dt: Timestep between samples in seconds, calculated as 1 / sample_rate.
        tau: Array of shape (n_mics, n_gridpts), storing the expected time
            delay between each microphone and each gridpoint.
        verbosity: Integer representing verbosity. Largely unused.
    
    Returns:
        A tuple (rsrp, a, xcorr_per_pair).

        rsrp: Array of shape (n_gridpts,) storing the calculated RSRP
            values at each gridpoint.
        a: Array of shape (n_mics,) storing the gain estimate for each
            microphone.
        xcorr_per_pair: Array of shape (n_gridpts, n_pairs) storing xcorr
            values at each grid point calculated from each pair.
    """
    N, n_mics = V.shape
    _, n_pts = tau.shape

    # determine the difference in expected time delays for each pair
    # of microphones (i, j) with i < j at each grid point.
    M = mixing_matrix_from_n_mics(n_mics)
    tau_diff = np.matmul(M, tau)

    # sum of squared fft samples for each mic
    V_ss_per_mic = np.sum(np.abs(V) ** 2, axis=0)

    # Gain estimate (RMS of V)
    a = np.sqrt(V_ss_per_mic) / N

    xcorr_raw, tau_line = xcorr_raw_from_dfted_clip(V, dt, M, verbosity)

    tau_diff_max = np.max(tau_diff)
    tau_diff_min = np.min(tau_diff)

    # if there are tau_diff values outside the range at which we
    # precomputed the cross-correlation, return NaN as we don't really
    # have enough information to determine the RSRP values.
    if (tau_diff_min < tau_line[0, 0]) or (tau_diff_max > tau_line[-1, -1]):
        nan_rsrp = np.full((n_pts,), np.nan)
        return nan_rsrp, a, None

    rsrp, x_corr_per_pair = rsrp_from_xcorr_raw_and_delta_tau(xcorr_raw, tau_line, tau_diff)

    return rsrp, a, x_corr_per_pair


def rsrp_grid_from_clip_and_xy_grids(
    v: np.ndarray,
    fs: int,
    f_lo: int,
    f_hi: int,
    temp: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    R: np.ndarray,
    verbosity: int
) -> Tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the RSRP values at each point of the grid defined by x_grid, y_grid.
    
    For more, see:
    https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/rsrp_grid_from_clip_and_xy_grids.m

    Args:
        v: Array of microphone signals. Expected shape: (N, n_mics), where
            N is the number of samples.
        fs: The sampling frequency of audio data, in Hz.
        f_lo: The lower bound of frequency band used, in Hz. Frequencies
            below this are zeroed after applying the FFT.
        f_hi: The upper bound of frequency band used, in Hz.
        temp: Air temperature at which data was collected, in degrees C.
        x_grid: Array storing the x-coordinate for every gridpoint at which
            to calculate RSRP values.
        y_grid: Array storing y-coordinate for every gridpoint.
        R: Array storing the position of each microphone in
            Cartesian coordinates. Expected shape: (3, n_mics)

    Returns:
        A tuple (rsrp_grid, a, vel, N_filt, V_filt, V, xcorr_per_pair_grid).

        rsrp_grid: RSRP values at each grid point on the arena floor,
            same shape as x_grid and y_grid.
        a: Array of shape (n_mics,) storing the gain estimate for each
            microphone.
        vel: Velocity of sound for the given air temperature, calculated
            by velocity_sound(temp).
        N_filt: Number of samples kept after filtering frequencies.
        V_filt: FFT'd audio input, with frequencies outside the range
            defined by f_lo and f_hi zeroed.
        V: Input argument V. Returned by MATLAB convention.
        xcorr_per_pair_grid: The xcorr values for each pair of microphones
            (i, j) with i < j at each gridpoint. The array will have shape:
            (x_grid.shape[0], x_grid.shape[1], n_pairs).
    """

    dt = 1 / fs
    N, n_mics = v.shape

    # In the original function, the authors plotted the clips on lines 20-33,
    # we'll skip that to avoid output clutter.

    # FFT input so we can filter frequencies
    V = np.fft.fft(v, axis=0)
    f = fft_base(N, fs / N)
    # Entries between the two frequencies
    keep_mask = ((f_lo <= np.abs(f)) & (np.abs(f) < f_hi)).ravel()

    # Zero frequencies outside our defined range
    V_filt = V.copy()
    V_filt[~keep_mask, :] = 0
    N_filt = np.sum(keep_mask)

    # create a grid of points spaced across the arena floor in 3d coordinates
    n_x, n_y = x_grid.shape
    n_r = n_x * n_y
    r_scan = np.zeros((3, 1, n_r), dtype=x_grid.dtype)
    r_scan[0] = x_grid.reshape((1, -1))
    r_scan[1] = y_grid.reshape((1, -1))

    # Calculate the individual coord differences between each grid point and each microphone
    # Note: the new axis allows the subtraction to be broadcast in the way bsxfun allowed
    # Specifically, (3, 1, n_r) - (3, 4, 1) --> (3, 4, n_r)
    rsubR = r_scan - R[..., np.newaxis]
    # Calculate Euclidean distance from the individual coordinate differences
    d = np.sqrt(np.sum(rsubR ** 2, axis=0))
    # Find the expected time delay for each microphone
    vel = velocity_sound(temp)
    tau = d / vel

    # calculate the rsrp + xcorr values at each gridpoint
    rsrp, a, xcorr_per_pair = rsrp_from_dfted_clip_and_delays_fast(V_filt, dt, tau, verbosity)

    # rsrp and x_corr_per_pair currently represent gridpoints as one axis
    # so expand those back out into a 2-axis grid-like array
    rsrp_grid = rsrp.reshape((n_x, n_y))
    n_pairs = xcorr_per_pair.shape[1]
    xcorr_per_pair_grid = xcorr_per_pair.reshape((n_x, n_y, n_pairs))

    return rsrp_grid, a, vel, N_filt, V_filt, V, xcorr_per_pair_grid


def xcorr_raw_from_dfted_clip(V: np.ndarray, dt: float, M: np.ndarray, verbosity=0):
    """
    Calculate the cross-correlation between signals recorded at each pair (i, j)
    of microphones with i < j, at every gridpoint on the arena floor.

    For more information, see:
    https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/xcorr_raw_from_dfted_clip.m

    Args:
        V: FFT'd result of mic data. Expected shape: (N, n_mics), where N is
            the number of samples.
        dt: Timestep between samples in seconds, calculated as 1 / sample_rate.
        M: Output of mixing_matrix_for_n_mics(n_mics).
    
    Returns:
        A tuple (xcorr_raw, tau_line).

        xcorr_raw: Array of shape (N*8, n_pairs) storing the cross-
            correlation values for each pair of microphones (i, j)
            with i < j, where the xcorr function is evaluated at each point
            of the frequency spectrum outputted by the FFT of V.
        tau_line: The values at which the xcorr was calculated. Expected
            shape: (N,)
    """

    # calculate the time lag for each element of xcorr_raw
    N, n_mics = V.shape
    # note: r is hardcoded in the original MUSE code
    r=8  # increase in sampling rate
    N_line=r*N
    # list of all values at which we calculate the cross-correlation
    tau_line= np.fft.fftshift(fft_base(N_line,dt/r))  #want large neg times first

    # calculate the cross power spectrum for each pair, show
    n_pairs = int(n_mics*(n_mics-1)/2)
    xcorr_raw = np.zeros((N_line,n_pairs))

    for i_pair in range(n_pairs):
        # find the values of i and j such that (i, j) corresponds to i_pair
        non_zero_idx = np.where(M[i_pair,:] != 0)[0]
        i_mike=non_zero_idx[0]
        j_mike=non_zero_idx[1]

        # calc cross-power spectrum
        Xcorr_raw_this = V[:,i_mike] * np.conj(V[:,j_mike]);

        # pad it, to increase resolution in time domain
        Xcorr_raw_this_padded = pad_at_high_freqs(Xcorr_raw_this, N_line);

        #  go to the time domain with IFFT, thus calculating the xcorr
        # between the two mics of the current pair, evaluated at every
        # frequency bin value from the FFT.
        xcorr_raw_this = np.fft.fftshift(np.real(np.fft.ifft(Xcorr_raw_this_padded)))

        #  store xcorrs
        xcorr_raw[:,i_pair] = xcorr_raw_this


    return xcorr_raw, tau_line


def argmax_grid(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    objective: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Return the x and y coordinates of the location with the maximum value
    of the objective, along with that attained maximum value.

    This function is used to calculate the location with the highest RSRP
    to form the sound source location estimate.
    
    For more, see:
    https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/argmax_grid.m
    
    Args:
        x_grid: array of shape (n_x, n_y) storing the x-coordinates of each
            gridpoint at which objective was calculated.
        y_grid: same as x_grid, but stores the y-coordinate.
        objective: array with the same shape as x_grid and y_grid, storing
            the value of some objective function at each gridpoint.
    
    Returns:
        A tuple (r_argmax, objective_max).

        r_argmax: Array of shape (2,1) storing the x and y coordinates of the
            estimated sound source location.
        objective_max: Maximum value of objective across the gridpoints.
    """
    i_min = np.unravel_index(np.argmax(objective), objective.shape)
    objective_max = objective[i_min]
    r_argmax = np.array([[x_grid[i_min]], [y_grid[i_min]]])
    return r_argmax, objective_max


def r_est_from_clip_simplified(
    v: np.ndarray,
    fs: int,
    f_lo: int,
    f_hi: int,
    temp: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    in_cage: Any,
    R: np.ndarray,
    verbosity: int
) -> Tuple[
        np.ndarray, float, np.ndarray, np.ndarray,
        float, int, np.ndarray, np.ndarray, np.ndarray
    ]:
    """
    Estimate sound source location from microphone array data.

    Specifically, calculate the Reduced Steered Response Power (RSRP)
    at a grid of points on the room floor, spaced by the provided resolution.
    Then, return the point with the max RSRP value as well as the grid itself.

    For more details, see:
    https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/r_est_from_clip_simplified.m

    Args:
        v: Array of microphone signals. Expected shape: (N, n_mics), where N
            is the number of samples.
        fs: The sampling frequency of audio data, in Hz.
        f_lo: The lower bound of frequency band used, in Hz. Frequencies
            below this are zeroed after applying the FFT.
        f_hi: The upper bound of frequency band used, in Hz.
        temp: Air temperature at which data was collected, in degrees C.
        x_grid: Array storing the x-coordinate for every gridpoint at which
            to calculate RSRP values.
        y_grid: Array storing y-coordinate for every gridpoint.
        in_cage: Unused argument. Kept for consistency with MATLAB.
        R: Array storing the position of each microphone in
            Cartesian coordinates. Expected shape: (3, n_mics)
        verbosity: Integer used to set verbosity level. Largely unused.

    Returns:
        A tuple (r_est, rsrp_max, rsrp_grid, a, vel, N_filt, V_filt, V, xcorr_per_pair_grid).
        
        r_est: Array of shape (2,1) storing the x and y coordinates of the
            estimated sound source location.
        rsrp_max: RSRP at location r_est; this is the max value across the grid.
        rsrp_grid: RSRP values at each grid point on the arena floor,
            same shape as x_grid and y_grid.
        a: Array of shape (n_mics,) storing the gain estimate for each
            microphone.
        vel: Velocity of sound for the given air temperature, calculated
            by velocity_sound(temp).
        N_filt: Number of samples kept after filtering frequencies.
        V_filt: FFT'd audio input, with frequencies outside the range
            defined by f_lo and f_hi zeroed.
        V: Input argument V. Returned by MATLAB convention.
        xcorr_per_pair_grid: The xcorr values for each pair of microphones
            (i, j) with i < j at each gridpoint. The array will have shape:
            (x_grid.shape[0], x_grid.shape[1], n_pairs). 
    """

    rsrp_grid,a,vel,N_filt,V_filt,V, xcorr_per_pair_grid = rsrp_grid_from_clip_and_xy_grids(
        v, fs, f_lo, f_hi, temp, x_grid, y_grid, R, verbosity
        )

    r_est, rsrp_max = argmax_grid(x_grid, y_grid, rsrp_grid)

    return r_est, rsrp_max, rsrp_grid, a, vel, N_filt, V_filt, V, xcorr_per_pair_grid
