"""
Code that performs neurostatistical analyses for research in the Brown lab.

"""

# ignore FutureWarning errors which clutter the outputs
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nitime.algorithms as alg
import nitime.utils as utils
import nitime.algorithms.spectral as tsa
import nitime as nit
from concurrent import futures
from itertools import islice
from scipy import signal, io, fftpack, stats
import scipy as sp
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def multitaper(data, movingwin, NW=3, adaptive=True, low_bias = True, jackknife=False, hz=250):
    """Multitaper spectrogram as performed by nitime. General rules of
    thumb:
    - NW is the normalized half-bandwidth of the data tapers. In MATLAB
      this is generally set to 3
    - adaptive is set to true, which means an adaptive weighting routine
      is used to combine the PSD estimates of different tapers.
    - low_bias is set to true which means Rather than use 2NW tapers, 
      only use the tapers that have better than 90% spectral concentration
      within the bandwidth (still using a maximum of 2NW tapers)

    Arguments:
    -----------
    data : np.ndarray
        in form [samples] (1D)
    movingwin : [winsize, winstep]
        i.e length of moving window and step size. This is in units of
        samples (so window/Fs is units time)
    NW : (optional)
        defaults to 3. maximum number of tapers is 2NW

    Note: jackknife is not currently returned
    """
    assert len(data.shape) == 1, "Data must be 1-D."
    assert len(movingwin) == 2, "Windowing must include size, step."

    # collect params
    winsize, winstep = movingwin
    windowed_data = window(data, winsize, winstep)

    mtsg = []
    for wd in windowed_data:
        f, psd_mt, jk = tsa.multi_taper_psd(
            np.asarray(wd), Fs=hz, NW=NW,
            adaptive=adaptive, jackknife=jackknife)
        mtsg.append(psd_mt)

    mtsg = np.vstack(mtsg).T
    t = np.arange(mtsg.shape[1]) * winstep / hz
    return mtsg, t, f


def notch_filter(data, f0, Fs, BW=3.):
    """Creates an iir notch filter to remove a sample frequency from the data.
    Arguments:
    -----------
    data : np.ndarray
        (in form samples x channels/trials)
    f0 : np.ndarray
        list of frequencies to exclude
    Fs : float
        Sampling frequency.
    BW : float (optional, defaults to 3.0Hz)
        -3dB bandwidth of the notch filter. Defaults to units of Fs.
    """

    w0 = f0 / (Fs / 2)  # Normalized Frequency
    bw = BW / (Fs / 2)  # Normalized Bandwidth
    Q = w0 / bw

    # Design notch filter
    b, a = signal.iirnotch(w0, Q)
    # Apply notch filter
    data_filt = signal.filtfilt(b, a, data)

    return data_filt


def butterworth_filter(data, Fs, lowcut, highcut, order=5):
    """Creates a butterworth bandpass filter to get specific signals.
    Arguments:
    -----------
    data : np.ndarray
        (in form samples x channels/trials)
    Fs : float
        Sampling frequency.
    lowcut : float
        Low frequency cutoff, set to 0 for lowpass
    highcut : float
        High frequency cutoff, set to np.infty for highpass
    order : int (optional, defaults to 5)
        Filter order.
    """

    nyq = 0.5 * Fs
    low = lowcut / nyq
    high = highcut / nyq
    if low == 0:
        b, a = signal.butter(order, high, btype='low')
    elif high == np.infty:
        b, a = signal.butter(order, low, btype='high')
    else:
        b, a = signal.butter(order, [low, high], btype='band')

    # Apply butterworth filter
    data_filt = signal.filtfilt(b, a, data)

    return data_filt

# functions for loading data


def read_timestamp(timestamp):
    '''
    timestamp: str
            with the format '2011-12-20 07:49:00.0' in army time

    returns time since the midnight in seconds as a float
    '''
    (date, clock) = timestamp.split(' ')
    (hour, minute, second) = clock.split(':')
    return 3600 * float(hour) + 60 * float(minute) + float(second)


def load_case_mat(filepath, hz=250, return_drug=False):
    """
    Extracts the data from one of the

    Arguments:
    -----------
    filepath : str
        Location of the file.

    Returns:
    ----------
    eegtimes, eegdata, effecttimes, consciousness, drugtimes, propofol
    """
    workspace = io.loadmat(filepath,
                           struct_as_record=False, squeeze_me=True)
    try:
        labels = workspace['HDR'].Label  # electrode labels
    except AttributeError:
        pass
    eegdata = workspace['data'][0, :]  # 0 is Fp1, 3 is F7
    eegtimes = np.arange(len(eegdata)) / hz
    return eegtimes, eegdata


def load_patrick2013mat(bhvr_file, eeg_file, hz=250, return_drug=False):
    """
    Extracts the behavioral data from one of the eeganesXX_laplac250_ch36.mat
    files. This file contains results from the behavioral analysis
    and time alginments. Based on Sourish matlab code.
    Extracts EEG data from channel 18 (approx Fp1) from correcponding eeganesXX.mat file

    Arguments:
    -----------
    bhvr_file : str
        Location of the file containing behavioral data.
    eeg_file : str
        Location of file containing eeg data from multiple leads including channel 18

    Returns:
    ----------
    eegtimes, eegdata, effecttimes, consciousness, drugtimes, propofol
    """
    # list to get which eeganes it is
    eeganes_list = np.array(
        ['02', '03', '04', '05', '07', '08', '09', '10', '13', '15'])

    # get the contents of the workspace
    workspace = io.loadmat(bhvr_file, struct_as_record=False, squeeze_me=True)
    aligntimes = workspace['aligntimes']
    bhvr = workspace['BHVR']
    drug = workspace['DRUG']    
    subject = bhvr.subject[-2:]
    subject_idx = np.where(eeganes_list == subject)[0][0]
    
    eeg_struct = io.loadmat(eeg_file, struct_as_record=False, squeeze_me=True)
    eegdata = eeg_struct['eegdata_ch18']
    # get the bhvr data
    T = bhvr.T
    prob_verbal_p500 = bhvr.prob_verbal.p500
    prob_burst_p500 = bhvr.prob_burst.p500
    prob_verbal_p025 = bhvr.prob_verbal.p025
    prob_burst_p025 = bhvr.prob_burst.p025
    prob_verbal_p975 = bhvr.prob_verbal.p975
    prob_burst_p975 = bhvr.prob_burst.p975
    if subject_idx == 0:
        # resize for eeganes02
        Tfull = T
        T = T[44:]
        prob_verbal_p500 = prob_verbal_p500[44:]
        prob_burst_p500 = prob_burst_p500[44:]
        prob_verbal_p025 = prob_verbal_p025[44:]
        prob_burst_p025 = prob_burst_p025[44:]
        prob_verbal_p975 = prob_verbal_p975[44:]
        prob_burst_p975 = prob_burst_p975[44:]

    pind = ~np.isnan(prob_verbal_p500)
    cind = ~np.isnan(prob_burst_p500)
    tb = T / 5000 / 60  # Time of behavioral events

    behaviour1 = prob_burst_p500
    behaviour2 = prob_burst_p500

    # Determine group-level E_on and E_off points
    i = 0
    timevector = []
    while len(timevector) < 30:
        if behaviour1[i] < 0.95:
            if i == 1:
                eontime = -10
                timevector = []
            elif behaviour1[i - 1] >= 0.95:
                eontime = tb[i]
                timevector = []
            timevector.append(tb[i])
        i += 1

    tempterp = tb[::-1]
    behaviour2 = behaviour1[::-1]

    i = 0
    timevector2 = []
    while len(timevector2) < 30:
        if behaviour2[i] < 0.95:
            if i == 1:
                eofftime = 500
                timevector2 = []
            elif behaviour2[i - 1] >= 0.95:
                eofftime = tempterp[i]
                timevector2 = []
            timevector2.append(tempterp[i])
        i += 1
    effecttimes = [eontime * 60, eofftime * 60]

    # eegtimes
    eegtimes = 1 / hz * np.arange(len(eegdata))

    if return_drug:
        drugtimes = drug.spump.T / drug.Fs
        return eegtimes, eegdata, effecttimes, aligntimes[subject_idx] * \
            60, drugtimes, drug.spump.prpfol
    else:
        return eegtimes, eegdata, effecttimes, aligntimes[subject_idx] * 60


# other functions

def butter_lowpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, [low], btype='low')
    return b, a


def butter_lowpass_filter(data, lowcut, fs, order=5):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def window_1(seq, n=2):
    """Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
       Window step is 1.
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def window(seq, winsize, winstep):
    """
    Returns a sliding window of width winsize and step of winstep
    from the data. Returns a list.
    """
    assert winsize >= winstep, "Window step must me at most window size."
    gen = islice(window_1(seq, n=winsize), None, None, winstep)
    for result in gen:
        yield list(result)


def coherogram(data, movingwin, NW=3, jackknife=False, hz=250, parallel=False,
               f_ub=50, f_lb=1):
    """Multitaper coherogram as performed by nitime. General rules of
    thumb:
    - NW is the normalized half-bandwidth of the data tapers. In MATLAB
      this is generally set to 3
    - low_bias is set to true, which means the number of tapers used is
      however many tapers have 90% spectral concentration within bandwidth
      with a max of 2NW

    Arguments:
    -----------
    data : np.ndarray
        in form [samples] (2D)
    movingwin : np.ndarray [winsize, winstep]
        i.e length of moving window and step size. This is in units of
        samples (so window/Fs is units time)
    NW : int (optional)
        defaults to 3. maximum number of tapers is 2NW
    parallel : bool (optional)
        defaults to false. chooses whether to use parallelization routine

    Note: jackknife is not currently returned
    """
    if parallel:
        # probably a better way to do this, shrugging emoji
        global tapers, eigs
        global nseq, K, L, sides, freq_idx

    assert len(data.shape) == 2, "Data must be 2-D for comparison."
    assert len(movingwin) == 2, "Windowing must include size, step."

    # set up the step size and slide
    winsize, winstep = movingwin
    n_samples = winsize
    nseq = data.shape[0]

    # get tapers and the eigenvalues with each
    # time-bandwidth and number of tapers
    K = 2 * NW - 1
    tapers, eigs = alg.dpss_windows(n_samples, NW, K)

    # get the number of calculations etc
    L = n_samples // 2 + 1
    sides = 'onesided'

    # get the frequency resolution
    TR = 1 / hz
    if L < n_samples:
        freqs = np.linspace(0, 1 / (2 * TR), L)
    else:
        freqs = np.linspace(0, 1 / TR, L, endpoint=False)

    # let's only look at freqs between
    freq_idx = np.where((freqs >= f_lb) * (freqs <= f_ub))[0]

    # get the interesting frequencies
    f = freqs[freq_idx]

    coh_t = []  # coherence matrices over time
    # dims: len(t), nseq, nseq, freq

    if parallel:

        with futures.ProcessPoolExecutor(max_workers=10) as executor:
            ex = executor.map(parallel_coh, window(data.T, winsize, winstep),
                              chunksize=100)

        for res in ex:
            coh_t.append(res)

        coh_t = np.array(coh_t)
    else:
        for di in window(data.T, winsize, winstep):
            # fix dimensions from window
            di = np.asarray(di).T

            # get the FT and the magnitudes of the squared power spectra
            tdata = tapers[None, :, :] * di[:, None, :]
            tspectra = fftpack.fft(tdata)
            ## mag_sqr_spectra = np.abs(tspectra)
            ## np.power(mag_sqr_spectra, 2, mag_sqr_spectra)

            w = np.empty((nseq, K, L))
            for i in range(nseq):
                w[i], _ = utils.adaptive_weights(tspectra[i], eigs,
                                                 sides=sides)

            csd_mat = np.zeros((nseq, nseq, L), 'D')
            psd_mat = np.zeros((2, nseq, nseq, L), 'd')
            coh_mat = np.zeros((nseq, nseq, L), 'd')
            coh_var = np.zeros_like(coh_mat)

            for i in range(nseq):
                for j in range(i):
                    sxy = alg.mtm_cross_spectrum(
                        tspectra[i], tspectra[j], (w[i], w[j]),
                        sides='onesided')

                    sxx = alg.mtm_cross_spectrum(
                        tspectra[i], tspectra[i], w[i], sides=sides)
                    syy = alg.mtm_cross_spectrum(
                        tspectra[j], tspectra[j], w[j], sides=sides)

                    psd_mat[0, i, j] = sxx
                    psd_mat[1, i, j] = syy

                    coh_mat[i, j] = np.abs(sxy) ** 2
                    coh_mat[i, j] /= (sxx * syy)
                    csd_mat[i, j] = sxy

                    # variance found from jackknife
                    #if i != j:
                    #    coh_var[i, j] = utils.jackknifed_coh_variance(
                    #        tspectra[i], tspectra[j], eigs, adaptive=True,)

            # normalize by number of tapers
            coh_mat_xform = utils.normalize_coherence(coh_mat, 2 * K - 2)

            # 95% CIs by jackknife variance calculation
            t025_limit = coh_mat_xform + \
                stats.distributions.t.ppf(.025, K - 1) * np.sqrt(coh_var)
            t975_limit = coh_mat_xform + \
                stats.distributions.t.ppf(.975, K - 1) * np.sqrt(coh_var)

            utils.normal_coherence_to_unit(t025_limit, 2 * K - 2, t025_limit)
            utils.normal_coherence_to_unit(t975_limit, 2 * K - 2, t975_limit)

            # only return area of interest
            coh = coh_mat[:, :, freq_idx]

            coh_t.append(coh)

    # reformat to array from list
    coh_t = np.array(coh_t)

    return f, coh_t


def parallel_coh(di):
    """
    Called above in parallel to construct the coherogram
    """
    # fix dimensions from window
    di = np.asarray(di).T

    # get the FT and the magnitudes of the squared power spectra
    tdata = tapers[None, :, :] * di[:, None, :]
    tspectra = fftpack.fft(tdata)
    ## mag_sqr_spectra = np.abs(tspectra)
    ## np.power(mag_sqr_spectra, 2, mag_sqr_spectra)

    w = np.empty((nseq, K, L))
    for i in range(nseq):
        w[i], _ = utils.adaptive_weights(tspectra[i], eigs, sides=sides)

    csd_mat = np.zeros((nseq, nseq, L), 'D')
    psd_mat = np.zeros((2, nseq, nseq, L), 'd')
    coh_mat = np.zeros((nseq, nseq, L), 'd')
    coh_var = np.zeros_like(coh_mat)

    for i in range(nseq):
        for j in range(i):
            sxy = alg.mtm_cross_spectrum(
                tspectra[i], tspectra[j], (w[i], w[j]), sides='onesided')

            sxx = alg.mtm_cross_spectrum(
                tspectra[i], tspectra[i], w[i], sides='onesided')
            syy = alg.mtm_cross_spectrum(
                tspectra[j], tspectra[j], w[j], sides='onesided')

            psd_mat[0, i, j] = sxx
            psd_mat[1, i, j] = syy

            coh_mat[i, j] = np.abs(sxy) ** 2
            coh_mat[i, j] /= (sxx * syy)
            csd_mat[i, j] = sxy

            # variance found from jackknife
            # if jackknife:
            #    if i != j:
            #        coh_var[i, j] = utils.jackknifed_coh_variance(
            #            tspectra[i], tspectra[j], eigs, adaptive=True,)

    # normalize by number of tapers
    coh_mat_xform = utils.normalize_coherence(coh_mat, 2 * K - 2)

    # 95% CIs by jackknife variance calculation
    t025_limit = coh_mat_xform + \
        stats.distributions.t.ppf(.025, K - 1) * np.sqrt(coh_var)
    t975_limit = coh_mat_xform + \
        stats.distributions.t.ppf(.975, K - 1) * np.sqrt(coh_var)

    utils.normal_coherence_to_unit(t025_limit, 2 * K - 2, t025_limit)
    utils.normal_coherence_to_unit(t975_limit, 2 * K - 2, t975_limit)

    coh = coh_mat[:, :, freq_idx]
    return coh


def quickplot_mtsg(t, f, S_db, flim=50, ax=None, hz=1, vmin=0, vmax=50,
                   plotsize=(3.5, 1), return_fig=False, cmap='magma'):
    """
    Quick plot utility for multitaper spectrogram
    """
    if ax is None:
        fig = plt.figure(figsize=plotsize)
        ax = plt.subplot()

    cbar = ax.pcolormesh(t / 60 / hz, f * hz,
                         S_db, cmap=cmap,
                         vmin=vmin, vmax=vmax)
    cbar.set_zorder(-10)
    cc = plt.colorbar(cbar)
    cc.set_label('Power (dB)')
    ax.set_ylim([0, flim])
    ax.set_ylabel('Frequency (Hz)')
    plt.tight_layout()
    if return_fig:
        return fig
