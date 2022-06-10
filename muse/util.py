from typing import Iterable, Tuple

import numpy as np


def velocity_sound(T):
    """
    From https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/velocity_sound.m

    This function calculates the velocity of sound based on the temp during recording.

    Formula for speed of sound in air taken from signals, sound and sensation (hartmann).

    T is the temp in Celsius, Vsound is the speed of sound in m/s.
    """
    Vsound = 331.3*np.sqrt(1+(T/273.16))
    return Vsound


def pad_at_high_freqs(x, n_padded):
    """ Ported from JaneliaSciComp/Muse.git
    See comments on https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/pad_at_high_freqs.m
    """
    num_elements = len(x)
    ratio = n_padded / num_elements
    num_nonnegative = int(np.ceil(num_elements / 2))
    num_negative = num_elements - num_nonnegative
    x_nonneg_freq = x[:num_nonnegative]
    x_neg_freq = x[num_nonnegative:]

    x_padded = np.zeros((n_padded,), dtype=x.dtype)
    x_padded[:num_nonnegative] = ratio * x_nonneg_freq
    x_padded[-num_negative:] = ratio * x_neg_freq
    return x_padded


def rsrp_from_xcorr_raw_and_delta_tau(xcorr_raw_all, tau_line, tau_diff):
    """ Ported from JaneliaSciComp/Muse.git
    See comments on https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/rsrp_from_xcorr_raw_and_delta_tau.m

    Argument types:
      - xcorr_raw_all: ndarray (N * 8, n_pairs)
      - tau_line: ndarray (N * 8, 1)
      - tau_diff: ndarray (n_pairs, n_r)

    Returns:
      - rsrp: ndarray (n_r,)
      - rsrp_per_pairs: ndarray (n_r, n_pairs)
    """
    n_pairs, n_r = tau_diff.shape
    # get interval at which we calculated cross-correlations
    tau_0 = tau_line[0, 0]
    dtau = (tau_line[-1, -1] - tau_0) / (len(tau_line) - 1)

    """
    rsrp_per_pairs = np.zeros((n_r, n_pairs), dtype=xcorr_raw_all.dtype)
    for i in range(n_r):
        for j in range(n_pairs):
            k_real = (tau_diff[j, i] - tau_0) / dtau + 1
            k_lo = np.floor(k_real)
            k_hi = k_lo + 1
            w_hi = k_real - k_lo
            rsrp_per_pairs[i, j] = \
                    (1 - w_hi) * xcorr_raw_all[int(k_lo), j] + \
                    w_hi * xcorr_raw_all[int(k_hi), j]
    # Originally, this was (1, n_r), here I've made its shape (n_r,)
    """

    # compute the cross-correlation for each tau_diff value by
    # interpolating linearly into the values we precomputed

    # find values in tau_line that sandwich each tau_diff
    k_real = ((tau_diff - tau_0) / dtau).T
    k_lo = np.floor(k_real).astype(int)
    k_hi = k_lo + 1
    w_hi = k_real - k_lo

    # There has to be a better way to do this
    _, j = np.indices(k_lo.shape)
    rsrp_per_pairs = (1 - w_hi) * xcorr_raw_all[k_lo, j] + w_hi * xcorr_raw_all[k_hi, j]
    rsrp = np.sum(rsrp_per_pairs, axis=1)
    return rsrp, rsrp_per_pairs


def mixing_matrix_from_n_mics(n_mics):
    """ Ported from JaneliaSciComp/Muse.git
    See comments on https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/mixing_matrix_from_n_mics.m

    Parameter types:
      - n_mics: int, scalar
    Returns:
      - M: ndarray: (n_pairs, n_mics), dtype=int, n_pairs = n_mics C 2 = (n_mics) * (n_mics - 1) / 2
    """
    # Note: I didn't implement iArrayUpper and jArrayUpper because they were never used outside this function
    # And the function could be implemented without them
    n_pairs = n_mics * (n_mics - 1) // 2
    if n_mics > 10:
        print(f'mixing_matrix_from_n_mics: There appear to be {n_mics} microphones. Perhaps V should be transposed?')
    M = np.zeros((n_pairs, n_mics), dtype=int)
    i = 0
    j = 1
    for row in range(n_pairs):
        M[row, i] = 1
        M[row, j] = -1
        j += 1
        if j >= n_mics:
            i += 1
            j = i + 1
    return M


def fft_base(N, dx):
    """
    From https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/fft_base.m

    Generates a frequency line to go with an N-point fft.
    Frequencies are in cycles per sample, i.e. they go from about -1/2 to about 1/2.

    In other words, given the number of samples (N) and the frequency domain bin width (dx),
    construct the range of frequency values associated with the fft results.
    """
    hi_x_sample_index = np.ceil(N/2).astype('int')
    x_pos = dx*np.linspace(0,hi_x_sample_index-1,hi_x_sample_index)
    x_neg = dx*np.linspace(-(N-hi_x_sample_index), -1, N-hi_x_sample_index)
    x = np.concatenate([x_pos, x_neg])[:, np.newaxis]

    return x


def rsrp_from_dfted_clip_and_delays_fast(V, dt, tau, verbosity):
    """ Ported from JaneliaSciComp/Muse.git
    See comments on https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/rsrp_from_dfted_clip_and_delays_fast.m

    Argument Types:
      - V: ndarray (N, n_mics), dtype=cfloat, N: number of time points
      - dt: float, scalar
      - tau: ndarray (n_mics, n_pts)
    """
    N, n_mics = V.shape
    _, n_pts = tau.shape

    M = mixing_matrix_from_n_mics(n_mics)
    tau_diff = np.matmul(M, tau)

    # sum of squared fft samples for each mic
    V_ss_per_mic = np.sum(np.abs(V) ** 2, axis=0)
    # Gain estimate (RMS of V)
    a = np.sqrt(V_ss_per_mic) / N

    xcorr_raw, tau_line = xcorr_raw_from_dfted_clip(V, dt, M, verbosity)

    tau_diff_max = np.max(tau_diff)
    tau_diff_min = np.min(tau_diff)

    if (tau_diff_min < tau_line[0, 0]) or (tau_diff_max > tau_line[-1, -1]):
        nan_rsrp = np.full((n_pts,), np.nan)
        return nan_rsrp, a, None

    rsrp, rsrp_per_pair = rsrp_from_xcorr_raw_and_delta_tau(xcorr_raw, tau_line, tau_diff)

    return rsrp, a, rsrp_per_pair


def rsrp_grid_from_clip_and_xy_grids(v, fs, f_lo, f_hi, temp, x_grid, y_grid, R, verbosity):
    """ Ported from JaneliaSciComp/Muse.git
    See comments on https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/rsrp_grid_from_clip_and_xy_grids.m

    Argument types:
      - v: Ndarray (N, n_mics), dtype=np.float; N is the number of time points
      - fs: float, scalar; The sample rate
      - f_lo: float, scalar; low end of bandpass filter
      - f_hi: float, scalar; top end of bandpass filter
      - temp: float, scalar; ambient temperature
      - x_grid: Ndarray (n_x, n_y); Shape determines n_r for rsrp
      - y_grid: Ndarray (n_x, n_y)
      - R: Ndarray (3, n_mics)
      - verbosity: int, scalar
    """

    dt = 1 / fs
    N, n_mics = v.shape

    # SKIPPED PLOTTING CODE
    # Lines 20-33 of original function

    # TODO: double check the axis here
    # Matlab performs the fft on each column independently, treating each column as a vector
    V = np.fft.fft(v, axis=0)
    f = fft_base(N, fs / N)
    # Entries between the two frequencies
    keep_mask = ((f_lo <= np.abs(f)) & (np.abs(f) < f_hi)).ravel()

    # TODO: Consider removing V as a return and correspondingly, perform operations directly on V
    # instead of copying to V_filt
    V_filt = V.copy()
    V_filt[~keep_mask, :] = 0
    N_filt = np.sum(keep_mask)

    # Note: Temp (original name) replaced with lowercase temp
    vel = velocity_sound(temp)

    # create a grid of points spaced across the arena floor
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
    tau = d / vel

    rsrp, a, rsrp_per_pair = rsrp_from_dfted_clip_and_delays_fast(V_filt, dt, tau, verbosity)

    rsrp_grid = rsrp.reshape((n_x, n_y))
    n_pairs = rsrp_per_pair.shape[1]
    rsrp_per_pair_grid = rsrp_per_pair.reshape((n_x, n_y, n_pairs))

    return rsrp_grid, a, vel, N_filt, V_filt, V, rsrp_per_pair_grid


def xcorr_raw_from_dfted_clip(V, dt, M, verbosity=0):
    """
    From https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/xcorr_raw_from_dfted_clip.m
    Calculate the cross-correlation between signals recorded at each pair (i, j) of microphones,
    where i < j.
    """

    # calculate the time lag for each element of xcorr_raw
    N,K = V.shape  # K the number of mikes
    # note: r is hardcoded in the original MUSE code
    r=8  # increase in sampling rate
    N_line=r*N
    # list of all values at which we calculate the cross-correlation
    tau_line= np.fft.fftshift(fft_base(N_line,dt/r))  #want large neg times first

    # calculate the cross power spectrum for each pair, show
    n_pairs = int(K*(K-1)/2)
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

        #  go to the time domain
        xcorr_raw_this = np.fft.fftshift(np.real(np.fft.ifft(Xcorr_raw_this_padded)))

        #  store xcorrs
        xcorr_raw[:,i_pair] = xcorr_raw_this


    return xcorr_raw, tau_line


def argmax_grid(x_grid, y_grid, objective):
    """ Ported from JaneliaSciComp/Muse.git
    See comments on https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/argmax_grid.m
    """
    i_min = np.unravel_index(np.argmax(objective), objective.shape)
    objective_max = objective[i_min]
    r_argmax = np.array([[x_grid[i_min]], [y_grid[i_min]]])
    return r_argmax, objective_max


def r_est_from_clip_simplified(v, fs, f_lo, f_hi, temp, x_grid, y_grid, in_cage, R, verbosity):
    """ Ported from JaneliaSciComp/Muse.git
    See comments on https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/r_est_from_clip_simplified.m

    Most arguments here are equivalent to those in rsrp_grid_from_clip_and_xy_grids
    """

    rsrp_grid,a,vel,N_filt,V_filt,V,rsrp_per_pair_grid = rsrp_grid_from_clip_and_xy_grids(v, fs, f_lo, f_hi, temp, x_grid, y_grid, R, verbosity)

    r_est, rsrp_max = argmax_grid(x_grid, y_grid, rsrp_grid)

    return r_est, rsrp_max, rsrp_grid, a, vel, N_filt, V_filt, V, rsrp_per_pair_grid


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
    r_estimates = []
    rsrp_grids = []

    N_MICS = v.shape[0]

    for i in range(N_MICS):
        # remove mic i from audio and mic position arrays
        v_omitted = np.delete(v, i, axis=0)
        mic_pos_omitted = np.delete(mic_positions, i, axis=0)
        # calculate estimates
        r_est, rsrp_grid = r_est_naive(
            v_omitted, fs, f_lo, f_hi, temp,
            x_len, y_len, resolution, mic_pos_omitted
        )
        r_estimates.append(r_est)
        rsrp_grids.append(rsrp_grid)
    
    avg_est = np.mean(r_estimates, axis=0)

    return avg_est, r_estimates, rsrp_grids

def make_xy_grid(x_len, y_len, resolution=0.00025):
    """
    Units in meters.

    Make x_grid and y_grid, given the dimensions of your arena.

    Resolution refers to the spatial resolution of your grid.

    """
    x_dim = int(x_len/resolution)
    y_dim = int(y_len/resolution)

    x_grid = (np.ones((x_dim, y_dim)) * np.linspace(0, x_len, x_dim).reshape((-1, 1))).T
    y_grid = (np.ones((x_dim, y_dim)) * np.linspace(0, y_len, y_dim).reshape((1, -1))).T

    return x_grid, y_grid
