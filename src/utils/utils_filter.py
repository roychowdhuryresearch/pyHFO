from scipy.signal import cheb2ord, cheby2, zpk2sos, sosfilt
import numpy as np

def filter_data(data, sos):
    filtered = sosfilt(sos, data)
    filtered = sosfilt(sos, np.flipud(filtered))
    filtered = np.flipud(filtered)
    return filtered

def construct_filter(fp, fs, rp, rs, space, sample_freq):
    # print("Constructing filter with fp: {}, fs: {}, rp: {}, rs: {}, space: {}, sample_freq: {}".format(fp, fs, rp, rs, space, sample_freq))
    if fs < 1 or rs < 1:
        raise ValueError("Invalid value for stop band.")
    
    filter_freq = [fp, fs]
    space = space

    nyq = sample_freq / 2
    # MGNM
    if fs >= .99 * nyq:
        fs = nyq * .99
    low, high = fp, fs

    scale = 0
    while 0 < low < 1:
        low *= 10
        scale += 1
    low = filter_freq[0] - (space * 10 ** (-1 * scale))

    scale = 0
    while high < 1:
        high *= 10
        scale += 1
    high = filter_freq[1] + (space * 10 ** (-1 * scale))  
    stop_freq = [low, high]
    # print([filter_freq[0] / nyq, filter_freq[1] / nyq], [stop_freq[0] / nyq, stop_freq[1] / nyq], rp,
    #                             rs)
    ps_order, wst = cheb2ord([filter_freq[0] / nyq, filter_freq[1] / nyq], [stop_freq[0] / nyq, stop_freq[1] / nyq], rp,
                                rs)
    z, p, k = cheby2(ps_order, rs, wst, btype='bandpass', analog=0, output='zpk')
    sos = zpk2sos(z, p, k)    
    return sos

if __name__ == "__main__":
    sos = construct_filter(60,500,0.5,93,0.535,2000)
    print(sos)
