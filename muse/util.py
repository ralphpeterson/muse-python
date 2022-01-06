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
    ''' Ported from JaneliaSciComp/Muse.git
    See comments on https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/pad_at_high_freqs.m
    '''
    num_elements = len(x)
    ratio = n_padded / num_elements
    num_nonnegative = np.ceil(num_elements / 2)
    num_negative = num_elements - num_nonnegative
    x_nonneg_freq = x[:num_nonnegative]
    x_neg_freq = x[num_nonnegative:]

    x_padded = np.zeros((n_padded,), dtype=x.dtype)
    x_padded[:num_nonnegative] = ratio * x_nonneg_freq
    x_padded[-num_negative:] = ratio * x_neg_freq
    return x_padded


def rsrp_from_xcorr_raw_and_delta_tau(xcorr_raw_all, tau_line, tau_diff):
    ''' Ported from JaneliaSciComp/Muse.git
    See comments on https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/rsrp_from_xcorr_raw_and_delta_tau.m

    Argument types:
      - xcorr_raw_all: ndarray (N, n_pairs)
      - tau_line: ndarray (N,)
      - tau_diff: ndarray (n_pairs, n_r)

    Returns:
      - rsrp: ndarray (n_r,)
      - rsrp_per_pairs: ndarray (n_r, n_pairs)
    '''
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

def fft_base(N, dx):
    """
    From https://github.com/JaneliaSciComp/Muse/blob/master/toolbox/fft_base.m
    
    Generates a frequency line to go with an N-point fft.  
    Frequencies are in cycles per sample, i.e. they go from about -1/2 to about 1/2.
    """
    hi_x_sample_index = np.ceil(N/2).astype('int')
    x_pos = dx*np.linspace(0,hi_x_sample_index-1,hi_x_sample_index)
    x_neg = dx*np.linspace(-(N-hi_x_sample_index), -1, N-hi_x_sample_index)
    x = np.array([x_pos, x_neg]);
    
    return x
