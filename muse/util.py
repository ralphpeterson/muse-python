import numpy as np
from scipy.fft import fft


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
      - xcorr_raw_all: ndarray (N, n_pairs)
      - tau_line: ndarray (N,)
      - tau_diff: ndarray (n_pairs, n_r)

    Returns:
      - rsrp: ndarray (n_r,)
      - rsrp_per_pairs: ndarray (n_r, n_pairs)
    """
    n_pairs, n_r = tau_diff.shape
    tau_0 = tau_line[0]
    dtau = (tau_line[-1] - tau_line[0]) / (len(tau_line) - 1)
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
    """
    hi_x_sample_index = np.ceil(N/2).astype('int')
    x_pos = dx*np.linspace(0,hi_x_sample_index-1,hi_x_sample_index)
    x_neg = dx*np.linspace(-(N-hi_x_sample_index), -1, N-hi_x_sample_index)
    x = np.array([x_pos, x_neg])

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

    if (tau_diff_min < tau_line[0]) or (tau_diff_max > tau_line[-1]):
        nan_rsrp = np.full((n_pts,), np.nan)
        return nan_rsrp, a, None

    rsrp, rsrp_per_pair = rsrp_from_xcorr_raw_and_delta_tau(xcorr_raw, tau_line, tau_diff)

    return rsrp, a, rsrp_per_pair


def rsrp_grid_from_clip_and_xy_grids(v, fs, f_lo, f_hi, temp, x_grid, y_grid, R, verbosity):
    """ Ported from JaneliaSciComp/Muse.git
    See comments on https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/rsrp_grid_from_clip_and_xy_grids.m

    Argument types:
      - v: Ndarray (N, n_mics), dtype=np.cfloat; N is the number of time points
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
    V = fft(v, axis=0)
    f = np.abs(fft_base(N, fs / N))
    # Entries between the two frequencies
    keep_mask = (f_lo <= f) & (f < f_hi)

    # TODO: Consider removing V as a return and correspondingly, perform operations directly on V
    # instead of copying to V_filt
    V_filt = V.copy()
    V_filt[~keep_mask, :] = 0
    N_filt = np.sum(keep_mask)

    # Note: Temp (original name) replaced with lowercase temp
    vel = velocity_sound(temp)

    n_x, n_y = x_grid.shape
    n_r = n_x * n_y
    r_scan = np.zeros((3, 1, n_r), dtype=x_grid.dtype)
    r_scan[0] = x_grid.reshape((1, -1))
    r_scan[1] = y_grid.reshape((1, -1))

    # The new axis allows the subtraction to be broadcast in the was bsxfun allowed
    rsubR = r_scan - R[..., np.newaxis]
    # Looks like a distance calculation
    d = np.sqrt(np.sum(rsubR ** 2, axis=0))
    tau = d / vel

    rsrp, a, rsrp_per_pair = rsrp_from_dfted_clip_and_delays_fast(V_filt, dt, tau, verbosity)

    rsrp_grid = rsrp.reshape((n_x, n_y))
    n_pairs = rsrp_per_pair.shape[1]
    rsrp_per_pair_grid = rsrp_per_pair.reshape((n_x, n_y, n_pairs))

    return rsrp_grid, a, vel, N_filt, V_filt, V, rsrp_per_pair_grid


def xcorr_raw_from_dfted_clip(V, dt, M, verbosity=0):
    """
    From https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/xcorr_raw_from_dfted_clip.m

    """

    # calculate the time lag for each element of xcorr_raw
    N,K = V.shape  # K the number of mikes
    r=8  # increase in sampling rate
    N_line=r*N
    tau_line= np.fft.fftshift(fft_base(N_line,dt/r))  #want large neg times first

    # calculate the cross power spectrum for each pair, show
    n_pairs = int(K*(K-1)/2)
    xcorr_raw = np.zeros((N_line,n_pairs))

    for i_pair in range(n_pairs):
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

