"""Analysis functions for testing FOOOF on simulated data."""

from collections import Counter

import numpy as np
# from fooof.analysis.periodic import get_band_peak, get_band_peak_fg

from settings import F_RANGE



"""Functions to analyze and investigate FOOOF results - periodic components."""

from fooof.core.items import PEAK_INDS

###################################################################################################
###################################################################################################


def get_band_peak(peak_params, band, select_highest=True, threshold=None, thresh_param='PW'):
    """Extract peaks within a given band of interest.

    Parameters
    ----------
    peak_params : 2d array
        Peak parameters, with shape of [n_peaks, 3].
    band : tuple of (float, float)
        Frequency range for the band of interest.
        Defined as: (lower_frequency_bound, upper_frequency_bound).
    select_highest : bool, optional, default: True
        Whether to return single peak (if True) or all peaks within the range found (if False).
        If True, returns the highest peak within the search range.
    threshold : float, optional
        A minimum threshold value to apply.
    thresh_param : {'PW', 'BW'}
        Which parameter to threshold on. 'PW' is power and 'BW' is bandwidth.

    Returns
    -------
    band_peaks : 1d or 2d array
        Peak data. Each row is a peak, as [CF, PW, BW].
    """

    # Return nan array if empty input
    if peak_params.size == 0:
        return np.array([np.nan, np.nan, np.nan])

    # Find indices of peaks in the specified range, and check the number found
    peak_inds = (peak_params[:, 0] >= band[0]) & (peak_params[:, 0] <= band[1])
    n_peaks = sum(peak_inds)

    # If there are no peaks within the specified range, return nan
    #   Note: this also catches and returns if the original input was empty
    if n_peaks == 0:
        return np.array([np.nan, np.nan, np.nan])

    band_peaks = peak_params[peak_inds, :]

    # Apply a minimum threshold, if one was provided
    if threshold:
        band_peaks = threshold_peaks(band_peaks, threshold, thresh_param)

    # If results > 1 and select_highest, then we return the highest peak
    #    Call a sub-function to select highest power peak in band
    if n_peaks > 1 and select_highest:
        band_peaks = get_highest_peak(band_peaks)

    # Squeeze so that if there is only 1 result, return single peak in flat array
    return np.squeeze(band_peaks)




def get_highest_peak(peak_params):
    """Extract the highest power peak.

    Parameters
    ----------
    peak_params : 2d array
        Peak parameters, with shape of [n_peaks, 3].

    Returns
    -------
    1d array
        Peak data. The row is a peak, as [CF, PW, BW].

    Notes
    -----
    This function returns the singular highest power peak from the input set, and as
    such is defined to work on periodic parameters from a single model fit.
    """

    # Catch & return NaN if empty
    if len(peak_params) == 0:
        return np.array([np.nan, np.nan, np.nan])

    high_ind = np.argmax(peak_params[:, 1])

    return peak_params[high_ind, :]




def threshold_peaks(peak_params, threshold, param='PW'):
    """Extract peaks that are above a given threshold value.

    Parameters
    ----------
    peak_params : 2d array
        Peak parameters, with shape of [n_peaks, 3] or [n_peaks, 4].
    threshold : float
        A minimum threshold value to apply.
    param : {'PW', 'BW'}
        Which parameter to threshold on. 'PW' is power and 'BW' is bandwidth.

    Returns
    -------
    thresholded_peaks : 2d array
        Peak parameters, with shape of [n_peaks, :].

    Notes
    -----
    This function can be applied to periodic parameters from an individual model,
    or a set or parameters from a group.
    """

    # Catch if input is empty & return nan if so
    if len(peak_params) == 0:
        return np.array([np.nan, np.nan, np.nan])

    # Otherwise, apply a mask to apply the requested threshold
    thresh_mask = peak_params[:, PEAK_INDS[param]] > threshold
    thresholded_peaks = peak_params[thresh_mask]

    return thresholded_peaks

###################################################################################################
###################################################################################################

def cohens_d(d1, d2):
    """Calculate cohens-D: (u1 - u2) / SDpooled."""

    return (np.mean(d1) - np.mean(d2)) / (np.sqrt((np.std(d1) ** 2 + np.std(d2) ** 2) / 2))


def calc_errors(truths, models, approach='abs'):
    """Calculate the error of model reconstructions with respect to ground truth.

    Error metrics available in `approach`: {'abs', 'sqrd'}
    """

    if approach == 'abs':
        errors = np.abs(truths - models)
    elif approach == 'sqrd':
        errors = (truths - models)**2
    else:
        raise ValueError('Approach not understood.')

    return errors


def get_ground_truth(sim_params):
    """Extract settings used to generated data (ground truth values)."""

    pe_truths = []
    ap_truths = []

    for ind, params in enumerate(sim_params):
        pe_truths.append([psd_params.periodic_params for psd_params in params])
        ap_truths.append([psd_params.aperiodic_params for psd_params in params])

    pe_truths = np.squeeze(np.array(pe_truths))
    ap_truths = np.array(ap_truths)

    return pe_truths, ap_truths


def get_fit_data(fgs, f_range=F_RANGE):
    """Extract fit results fit to simulated data."""

    # Extract data of interest from FOOOF fits
    peak_fits = []; ap_fits = []; err_fits = []; r2_fits = []; n_peaks = []

    for fg in fgs:
        # peak_fits.append(get_band_peak_fg(fg, f_range, attribute='gaussian_params'))

        peaks = np.empty((0, 3))
        for f_res in fg:  
            peaks = np.vstack((peaks, get_band_peak(f_res.gaussian_params, f_range, select_highest=True)))
        peak_fits.append(peaks)
        
        ap_fits.append(fg.get_params('aperiodic_params'))
        err_fits.append(fg.get_params('error'))
        r2_fits.append(fg.get_params('r_squared'))
        n_peaks.append(fg.n_peaks_)

    peak_fits = np.array(peak_fits)
    ap_fits = np.array(ap_fits)
    err_fits = np.array(err_fits)
    r2_fits = np.array(r2_fits)
    n_peaks = np.array(n_peaks)

    return peak_fits, ap_fits, err_fits, r2_fits, n_peaks


def count_peak_conditions(n_fit_peaks, conditions):
    """Count the number of fit peaks, across simulated conditions."""

    # Grab all flattened data for conditions
    conds = np.array([[nn] * n_fit_peaks.shape[1] for nn in conditions]).flatten()

    # Grab data for number of peaks fit
    n_fit_peaks = n_fit_peaks.flatten()

    # Collect together # simulated & # fit, for plotting
    data = []
    for aa, bb in zip(conds, n_fit_peaks):
        data.append((aa, bb))
    n_peak_counter = Counter(data)

    return n_peak_counter


def harmonic_mapping(fg):
    """Get all peaks from a FOOOFGroup and compute harmonic mapping on the CFs."""

    f_mapping = []
    for f_res in fg:
        cfs = f_res.peak_params[:, 0]
        if len(cfs) > 0:
            f_mapping.append(list(cfs / cfs[0]))

    return f_mapping
