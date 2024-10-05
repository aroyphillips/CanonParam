import numpy as np
import fooof
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
import importlib
import mne
import pandas as pd
import seaborn as sns
import CanonParam._temporary_data_loading.load_open_closed_data as locd
import CanonParam._temporary_data_loading.data_utils as du
import CanonParam._temporary_data_loading.epoch_creation as ec

import CanonParam.parameterize_spectra as bandparams


def extract_param_spectra(freqs, power_spectrum, bands='standard', aperiodic_mode='knee', l_freq=0.3, h_freq=250, log_freqs=True):
    param_spectra = bandparams.ParamSpectra(bands=bands, log_freqs=log_freqs, n_division=1, l_freq=l_freq, h_freq=h_freq, prominence=0.5, linenoise=60, aperiodic_mode=aperiodic_mode, verbose=0)
    param_spectra._error_metric = 'RMSE'
    param_spectra.fit(freqs, power_spectrum)
    return param_spectra

def extract_fooof_param_spectra(freqs, power_spectrum, bands='standard', aperiodic_mode='knee', l_freq=0.3, h_freq=250, log_freqs=True):
    max_n_peaks = len(bands) if type(bands) == list else 7
    param_spectra = fooof.FOOOF(peak_width_limits=(2.0, 100.0), max_n_peaks=max_n_peaks, min_peak_height=0.2, peak_threshold=2.0, aperiodic_mode='knee', verbose=False)
    param_spectra._error_metric = 'RMSE'
    param_spectra.fit(freqs, power_spectrum)
    return param_spectra

def plot_param_spectra(freqs, power_spectrum, param_spectra=None, bands='standard', channel='', aperiodic_mode='knee', l_freq=0.3, h_freq=250, log_freqs=True):
    if param_spectra is None:
        param_spectra = extract_param_spectra(freqs, power_spectrum, bands=bands, aperiodic_mode=aperiodic_mode, l_freq=l_freq, h_freq=h_freq, log_freqs=log_freqs)
    print(param_spectra.get_params_out())
    plt.figure()
    plt.plot(freqs, np.log10(power_spectrum), label='original spectrum')
    plt.plot(freqs, param_spectra.modeled_spectrum_, label='parameterized spectrum')
    plt.plot(freqs, param_spectra._ap_fit, label='aperiodic component')
    plt.plot(freqs, param_spectra._peak_fit+np.mean(param_spectra._ap_fit), label='periodic component')
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.title(f'Parameterized spectra using {bands if type(bands) == str else "custom"} bands {channel}')
    plt.show()
    

def extract_subject_params(subj, psddict, bands, aperiodic_mode, l_freq, h_freq, log_freqs, sdx=0, n_subjs=100, normalize_psd=False, normalize_method='median', channel_independence=False, normalize_a=None, normalize_b=None, channels=['C3','C4']):
    """
    Finds the bandparams for a given subject
    args:
        sdx: index for printing
        subj: subj id
        psddict: dict containing psd
        bands: list of tups
        aperiodic_mode: 'fixed' or knee'
        l_freq: lower bound
        h_freq: upper_boud
        log_freqs: whether to find guassians from log form of freqs
        normalize_psd: whether to normalize
        normalize_method: median, mean, minmax, robust, z_score
        normalize_a: None for subject independent scaling or (middle value or min value for zscore/robust or minmax)
        normalize_b: None for subject independent scaling or (range value or max value for zscore/robust or minmax)
        channels:
    """
    print(f"Starting {subj} ({sdx}/{n_subjs})")
    st1 = time.time()
    subj_params_out = {}
    times2 = []
    times3 = []
    for stdx, state in enumerate(list(psddict.keys())):
        st2 = time.time()
        subj_params_out[state] = {}
        psd, freqs = psddict[state].get_data(return_freqs=True)
        if normalize_psd:
            print(f"Normalizing psd with: {normalize_method}, {channel_independence}, {normalize_a}, {normalize_b}")
            psd = du.PSDScaler(normalize_method=normalize_method, channel_independence=channel_independence, normalize_a=normalize_a, normalize_b=normalize_b).fit_transform(psd)

        assert len(channels) == psd.shape[1], f"Expected {len(channels)} channels, got {psd.shape[1]} out of {psd.shape}"

        mean_psd = np.mean(psd, axis=0)
        for chdx in range(len(channels)):
            st3 = time.time()
            ch_psd = mean_psd[chdx]
            subj_params_out[state][channels[chdx]] = extract_param_spectra(freqs, ch_psd, bands=bands, aperiodic_mode=aperiodic_mode, l_freq=l_freq, h_freq=h_freq, log_freqs=log_freqs).get_params_out()
            et3 = time.time()
            times3.append(et3-st3)
            print(f"Finished channel {channels[chdx]} ({chdx}/{len(channels)}) in {et3-st3} seconds for state {state} ({stdx+1}/{len(psddict.keys())}) for subj {subj} ({sdx}/{n_subjs})")
        et2 = time.time()
        times2.append(et2-st2)
        print(f"Finished state {state} ({stdx}/{n_subjs}) in {et2-st2} seconds for subj {subj} ({sdx+1}/{n_subjs})")
    et1 = time.time()
    print(f"Finished {subj} ({sdx}/{len(psddict.keys())}) in {et1-st1} seconds")
    return subj, subj_params_out, et1-st1, times2, times3


def extract_subject_fooof_params(subj, psddict, bands, aperiodic_mode, l_freq, h_freq, log_freqs, sdx=0, n_subjs=100, normalize_psd=False, normalize_method='median', channel_independence=False, normalize_a=None, normalize_b=None, channels=['C3','C4']):
    """
    Finds the FOOOF bandparams for a given subject
    args:
        sdx: index for printing
        subj: subj id
        psddict: dict containing psd
        bands: list of tups
        aperiodic_mode: 'fixed' or knee'
        l_freq: lower bound
        h_freq: upper_boud
        log_freqs: whether to find guassians from log form of freqs
        normalize_psd: whether to normalize
        normalize_method: median, mean, minmax, robust, z_score
        normalize_a: None for subject independent scaling or (middle value or min value for zscore/robust or minmax)
        normalize_b: None for subject independent scaling or (range value or max value for zscore/robust or minmax)
        channels:
    """
    print(f"Starting {subj} ({sdx}/{n_subjs})")
    st1 = time.time()
    subj_params_out = {}
    times2 = []
    times3 = []
    for stdx, state in enumerate(list(psddict.keys())):
        st2 = time.time()
        subj_params_out[state] = {}
        psd, freqs = psddict[state].get_data(return_freqs=True)
        if normalize_psd:
            print(f"Normalizing psd with: {normalize_method}, {channel_independence}, {normalize_a}, {normalize_b}")
            psd = du.PSDScaler(normalize_method=normalize_method, channel_independence=channel_independence, normalize_a=normalize_a, normalize_b=normalize_b).fit_transform(psd)

        assert len(channels) == psd.shape[1], f"Expected {len(channels)} channels, got {psd.shape[1]} out of {psd.shape}"
        mean_psd = np.mean(psd, axis=0)
        for chdx in range(len(channels)):
            st3 = time.time()
            ch_psd = mean_psd[chdx]
            freqs, ch_psd = process_power_spectrum(freqs, ch_psd)
            fitted_fooof = extract_fooof_param_spectra(freqs, ch_psd, bands=bands, aperiodic_mode=aperiodic_mode, l_freq=l_freq, h_freq=h_freq, log_freqs=log_freqs)
            try:
                ap_params = fitted_fooof.get_params('aperiodic_params')
                peak_params = fitted_fooof.get_params('peak_params')
                gaus_params = fitted_fooof.get_params('gaussian_params')
                error = fitted_fooof.get_params('error')
                r_squared = fitted_fooof.get_params('r_squared')
            except:
                ap_params = None
                peak_params = None
                gaus_params = None
                error = None
                r_squared = None

            subj_params_out[state][channels[chdx]] = {'aperiodic_params': ap_params, 'peak_params': peak_params, 'gaussian_params': gaus_params, 'error': error, 'r_squared': r_squared}
            et3 = time.time()
            times3.append(et3-st3)
            print(f"Finished channel {channels[chdx]} ({chdx}/{len(channels)}) in {et3-st3} seconds for state {state} ({stdx+1}/{len(psddict.keys())}) for subj {subj} ({sdx}/{n_subjs})")
        et2 = time.time()
        times2.append(et2-st2)
        print(f"Finished state {state} ({stdx}/{n_subjs}) in {et2-st2} seconds for subj {subj} ({sdx+1}/{n_subjs})")
    et1 = time.time()
    print(f"Finished {subj} ({sdx}/{len(psddict.keys())}) in {et1-st1} seconds")
    return subj, subj_params_out, et1-st1, times2, times3

def extract_all_params_out_subj_level(psddict, bands='standard', n_jobs=19, aperiodic_mode='fixed', l_freq=0.3, h_freq=250, log_freqs=True, channels=['C3','C4'], normalize_psd=False, normalize_method='median', channel_independence=False, normalize_a=None, normalize_b=None):
    all_params_out = {}
    results = Parallel(n_jobs=n_jobs)(delayed(extract_subject_params)(subj, psddict[subj], bands, aperiodic_mode, l_freq, h_freq, log_freqs, sdx=sdx, n_subjs=len(psddict.keys()), channels=channels, normalize_psd=normalize_psd, normalize_method=normalize_method, channel_independence=channel_independence, normalize_a=normalize_a, normalize_b=normalize_b) for sdx, subj in enumerate(list(psddict.keys())))
    times = []
    times2 = []
    times3 = []
    for subj, subj_params_out, time1, subj_times2, subj_times3 in results:
        all_params_out[subj] = subj_params_out
        times.append(time1)
        times2.extend(subj_times2)
        times3.extend(subj_times3)
    print('Mean time per channel:', np.mean(times3))
    return all_params_out



def extract_all_fooof_params_out_subj_level(psddict, bands='standard', n_jobs=19, aperiodic_mode='fixed', l_freq=0.3, h_freq=250, log_freqs=True, channels=['C3','C4'], normalize_psd=False, normalize_method='median', channel_independence=False, normalize_a=None, normalize_b=None):
    all_params_out = {}
    results = Parallel(n_jobs=n_jobs)(delayed(extract_subject_fooof_params)(subj, psddict[subj], bands, aperiodic_mode, l_freq, h_freq, log_freqs, sdx=sdx, n_subjs=len(psddict.keys()), channels=channels, normalize_psd=normalize_psd, normalize_method=normalize_method, channel_independence=channel_independence, normalize_a=normalize_a, normalize_b=normalize_b) for sdx, subj in enumerate(list(psddict.keys())))
    times = []
    times2 = []
    times3 = []
    for subj, subj_params_out, time1, subj_times2, subj_times3 in results:
        all_params_out[subj] = subj_params_out
        times.append(time1)
        times2.extend(subj_times2)
        times3.extend(subj_times3)
    print('Mean time per channel:', np.mean(times3))
    return all_params_out


def get_bp_dfs(all_params_out, bandnames, bands, aperiodic_mode, channels=['C3', 'C4']):
    subjs = list(all_params_out.keys())
    all_rsq = np.empty((len(subjs), 2*len(channels)))
    all_error = np.empty((len(subjs), 2*len(channels)))
    if aperiodic_mode == 'fixed':
        ap_feats = ['offset', 'exponent']
    elif aperiodic_mode == 'knee':
        ap_feats = ['offset', 'knee', 'exponent']
    peak_feats= ['center', 'height', 'width']
    all_aperiodic = np.empty((len(subjs), 2*len(channels)*len(ap_feats)))
    peak_params = np.empty((len(subjs), 2*len(channels)*len(bandnames)*len(peak_feats)))
    n_feats_total = len(ap_feats)+len(bandnames)*len(peak_feats)
    full_tensor = np.empty((len(subjs), len(channels), 2*n_feats_total))
    for sdx, subj in enumerate(subjs):
        for stdx, state in enumerate(['open', 'closed']):
            for chdx, channel in enumerate(channels):
                all_rsq[sdx, stdx*len(channels)+chdx] = all_params_out[subj][state][channel]['r_squared']
                all_error[sdx, stdx*len(channels)+chdx] = all_params_out[subj][state][channel]['error']
                all_aperiodic[sdx, stdx*len(channels)*len(ap_feats)+chdx*len(ap_feats):stdx*len(channels)*len(ap_feats)+chdx*len(ap_feats)+len(ap_feats)] = all_params_out[subj][state][channel]['aperiodic_params']
                peak_params[sdx, stdx*len(channels)*len(bandnames)*len(peak_feats)+chdx*len(bandnames)*len(peak_feats):stdx*len(channels)*len(bandnames)*len(peak_feats)+chdx*len(bandnames)*len(peak_feats)+len(bandnames)*len(peak_feats)] = all_params_out[subj][state][channel]['peak_params']
                # full_tensor[sdx, stdx*len(channels)+chdx, 0] = viz_params[subj][state][channel]['r_squared']
                # full_tensor[sdx, stdx*len(channels)+chdx, 1] = viz_params[subj][state][channel]['error']
                # full_tensor[sdx, stdx*len(channels)+chdx, 2:2+len(ap_feats)] = viz_params[subj][state][channel]['aperiodic_params']
                # full_tensor[sdx, stdx*len(channels)+chdx, 2+len(ap_feats):] = viz_params[subj][state][channel]['peak_params']
                full_tensor[sdx, chdx, stdx*n_feats_total:stdx*n_feats_total+n_feats_total] = np.concatenate([all_params_out[subj][state][channel]['aperiodic_params'], all_params_out[subj][state][channel]['peak_params']])


    all_rsq_df = pd.DataFrame(all_rsq, columns=[f'{channel}_{state}_rsq' for state in ['open', 'closed'] for channel in channels], index=subjs)
    all_error_df = pd.DataFrame(all_error, columns=[f'{channel}_{state}_mse' for state in ['open', 'closed'] for channel in channels], index=subjs)
    all_aperiodic_df = pd.DataFrame(all_aperiodic, columns=[f'{channel}_{state}_{param}' for state in ['open', 'closed'] for channel in channels for param in ap_feats], index=subjs)
    all_peak_params_df = pd.DataFrame(peak_params, columns=[f'{channel}_{state}_{band}_{param}' for state in ['open', 'closed'] for channel in channels for band in bandnames for param in peak_feats], index=subjs)
    all_param_df = pd.concat([all_rsq_df, all_error_df, all_aperiodic_df, all_peak_params_df], axis=1)
    tensor_axis1 = [f'{channel}_{state}' for state in ['open', 'closed'] for channel in channels]
    tensor_axis2 = ['rsq', 'error', 'offset', 'knee', 'exponent'] + [f'{band}_{param}' for band in bandnames for param in ['center', 'height', 'width']]
    
    out_dict = {
        'rsq': all_rsq_df,
        'error': all_error_df,
        'aperiodic': all_aperiodic_df,
        'peak_params': all_peak_params_df,
        'all_param': all_param_df,
        'full_tensor': full_tensor,
        'tensor_axis1': tensor_axis1,
        'tensor_axis2': tensor_axis2,
        
    }
    return out_dict

def get_fooof_bp_dfs(all_params_out, max_n_peaks, aperiodic_mode, channels=['C3', 'C4']):
    subjs = list(all_params_out.keys())
    all_rsq = np.empty((len(subjs), 2*len(channels)))
    all_error = np.empty((len(subjs), 2*len(channels)))
    if aperiodic_mode == 'fixed':
        ap_feats = ['offset', 'exponent']
    elif aperiodic_mode == 'knee':
        ap_feats = ['offset', 'knee', 'exponent']
    peak_feats= ['center', 'height', 'width']
    all_aperiodic = np.empty((len(subjs), 2*len(channels)*len(ap_feats)))
    peak_params = np.empty((len(subjs), 2*len(channels)*max_n_peaks*len(peak_feats)))
    n_feats_total = len(ap_feats)+max_n_peaks*len(peak_feats)
    full_tensor = np.empty((len(subjs), len(channels), 2*n_feats_total))
    
    all_aperiodic.fill(np.nan)
    peak_params.fill(np.nan)
    full_tensor.fill(np.nan)
    for sdx, subj in enumerate(subjs):
        for stdx, state in enumerate(['open', 'closed']):
            for chdx, channel in enumerate(channels):

                rsq_idx = stdx*len(channels)+chdx
                error_idx = stdx*len(channels)+chdx
                ap_sidx = stdx*len(channels)*len(ap_feats)+chdx*len(ap_feats)# the start index for aperiodic features
                peak_sidx = stdx*len(channels)*max_n_peaks*len(peak_feats)+chdx*max_n_peaks*len(peak_feats) # the start index for peak features
                if all_params_out[subj][state][channel]['aperiodic_params'] is None:
                    continue
                else:
                    all_rsq[sdx, rsq_idx] = all_params_out[subj][state][channel]['r_squared']
                    all_error[sdx, error_idx] = all_params_out[subj][state][channel]['error']
                    all_aperiodic[sdx, ap_sidx:ap_sidx+len(ap_feats)] = all_params_out[subj][state][channel]['aperiodic_params']
                    curr_peak_params = all_params_out[subj][state][channel]['peak_params']
                    peak_params[sdx, peak_sidx:peak_sidx+len(curr_peak_params.flatten())] = curr_peak_params.flatten()
                    full_tensor[sdx, chdx, stdx*n_feats_total:stdx*n_feats_total+len(ap_feats)+len(curr_peak_params.flatten())] = np.concatenate([all_params_out[subj][state][channel]['aperiodic_params'], all_params_out[subj][state][channel]['peak_params'].flatten()])

    all_rsq_df = pd.DataFrame(all_rsq, columns=[f'{channel}_{state}_rsq' for state in ['open', 'closed'] for channel in channels], index=subjs)
    all_error_df = pd.DataFrame(all_error, columns=[f'{channel}_{state}_mse' for state in ['open', 'closed'] for channel in channels], index=subjs)
    all_aperiodic_df = pd.DataFrame(all_aperiodic, columns=[f'{channel}_{state}_{param}' for state in ['open', 'closed'] for channel in channels for param in ap_feats], index=subjs)
    all_peak_params_df = pd.DataFrame(peak_params, columns=[f'{channel}_{state}_{pk}_{param}' for state in ['open', 'closed'] for channel in channels for pk in range(max_n_peaks) for param in peak_feats], index=subjs)
    all_param_df = pd.concat([all_rsq_df, all_error_df, all_aperiodic_df, all_peak_params_df], axis=1)
    tensor_axis1 = [f'{channel}_{state}' for state in ['open', 'closed'] for channel in channels]
    tensor_axis2 = ['rsq', 'error', 'offset', 'knee', 'exponent'] + [f'peak{pk}_{param}' for pk in range(max_n_peaks) for param in ['center', 'height', 'width']]
    
    out_dict = {
        'rsq': all_rsq_df,
        'error': all_error_df,
        'aperiodic': all_aperiodic_df,
        'peak_params': all_peak_params_df,
        'all_param': all_param_df,
        'full_tensor': full_tensor,
        'tensor_axis1': tensor_axis1,
        'tensor_axis2': tensor_axis2,
        
    }
    return out_dict


def reconstruct_bandparam_spectrum(freqs, aperiodic_params, gaussian_params, bands):
    aperiodic = bandparams.gen_aperiodic(freqs, aperiodic_params)
    peaks = bandparams.sum_of_gaussians(np.log(freqs), bands, np.ndarray.flatten(gaussian_params))
    return aperiodic + peaks, peaks, aperiodic


def process_power_spectrum(freqs, power_spectrum, linenoise=60, prominence=0.5):

    max_harmonic=int(freqs.max()//linenoise)
    if max_harmonic > 0:
        harmonics = [60*int(ii) for ii in range(1, max_harmonic+1)]
        noise_pks, noise_ranges = bandparams.detect_powerline_harmonics_peak_widths(freqs, power_spectrum, harmonics=harmonics, prominence=prominence)
        if len(noise_pks) > 0:
            for noise_range in noise_ranges:
                freqs, power_spectrum = fooof.utils.interpolate_spectrum(np.copy(freqs), np.copy(power_spectrum), interp_range=noise_range)

    return freqs, power_spectrum

def reconstruct_fooof_spectrum(freqs, aperiodic_params, gaussian_params):
    aperiodic = fooof.objs.fit.gen_aperiodic(freqs, aperiodic_params)
    peaks = fooof.objs.fit.gen_periodic(freqs, np.ndarray.flatten(gaussian_params))
    return aperiodic + peaks, peaks, aperiodic


def main(n_jobs=1, tables_folder='data/tables/'):
    data_params = {
    'num_subjs': 151,
    'l_freq': 0.3,
    'h_freq': None,
    'fs_baseline': 500,
    'order': 6,
    'notches': [],
    'notch_width': [],
    'reference_method': 'CSD',
    'filter_ecg': False,
    'late_filter_ecg': False
    }
    epochs_params = {
    'window_len': 10,
    'overlap': 0.1,
    'epochs_per_subject': 10,
    'epoch_method': 'random',
    }

    normalization_kwargs = {
        'normalize_epochs': True,
        'normalize_psd': False,
        'norm_epochs_scalings': 'median', # 'median', 'mean'
        'norm_psd_scalings': 'median', # 'median', 'mean'
    }

    epochs_dict = ec.main_make_epochs(data_params, **epochs_params, which_split='train', n_jobs=n_jobs, tables_folder=tables_folder) # ~2 Gigabytes'
    common_channels = ec.get_common_channels(epochs_dict)


    # drop the ecg chanel
    eeg_common_channels = du.isolate_eeg_channels(common_channels)
    # feature extraction / transformation
    subjs = list(epochs_dict.keys())

    if normalization_kwargs['normalize_epochs']:
        epochs_scaler = mne.decoding.Scaler(scalings=normalization_kwargs['norm_epochs_scalings'])
        epochs_dict_picks = {subj: {state: epochs_dict[subj][state].copy().pick_channels(eeg_common_channels) for state in ['open', 'closed']} for subj in subjs}
        scaled_epochs_dict = {subj: {state: epochs_scaler.fit_transform(epochs_dict_picks[subj][state].get_data(picks=eeg_common_channels)) for state in ['open', 'closed']} for subj in subjs}
        info_dict = {subj: {state: epochs_dict[subj][state].info for state in ['open', 'closed']} for subj in subjs}
        # change the channels in the info_dict
        epochs_dict = {subj: {state: mne.EpochsArray(scaled_epochs_dict[subj][state], info_dict[subj][state], tmin=epochs_dict[subj][state].tmin, verbose=0) for state in ['open', 'closed']} for subj in subjs}

    kwargs = {'fs_baseline': 500}
    st = time.time()
    psd_method = 'multitaper'
    fmin = data_params['l_freq']
    fmax = data_params['h_freq'] if data_params['h_freq'] is not None else kwargs['fs_baseline']//2
    psd_dict = {subj: {state: epochs_dict[subj][state].compute_psd(fmin=fmin, fmax=fmax, method=psd_method, picks=eeg_common_channels) for state in ['open', 'closed']} for subj in subjs} # fmin, fmax # 2 min

    psdtime = time.time()-st

    print(f"PSD took {psdtime} seconds")

    param_spectra_params = {
    'bands': 'standard_nohigh',
    'aperiodic_mode': 'knee',
    'l_freq': 0.3,
    'h_freq': 250,
    'log_freqs': True,
    }

    psd_normalize_kwargs = {
        'normalize_psd': True,
        'normalize_method': 'median',
        'channel_independence': False,
        'normalize_a': None,
        'normalize_b': None,
    }
    st = time.time()
    all_parameterized_spectra = extract_all_params_out_subj_level(psd_dict, channels=eeg_common_channels, n_jobs=8, **param_spectra_params, **psd_normalize_kwargs) 
    paramtime = time.time()-st


    print(f"Finished running our method in {time.time() -st} seconds")
    standard_bands = [(0.3, 1.5), (1.5, 4), (4, 8), (8, 12.5), (12.5, 30), (30, 70), (70, 150)]
    bandnames = ['low delta', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'chi']
    param_dfs = get_bp_dfs(all_parameterized_spectra, bandnames=bandnames, bands=standard_bands, aperiodic_mode=param_spectra_params['aperiodic_mode'], channels=eeg_common_channels)


    # fit their model
    st = time.time()
    all_fooof_parameterized_spectra = extract_all_fooof_params_out_subj_level(psd_dict, channels=eeg_common_channels, n_jobs=8, **param_spectra_params, **psd_normalize_kwargs) 
    foooftime = time.time()-st

    print(f"Finished running all FOOOF in {time.time() -st} seconds")
    standard_bands = [(0.3, 1.5), (1.5, 4), (4, 8), (8, 12.5), (12.5, 30), (30, 70), (70, 150)]
    bandnames = ['low delta', 'delta', 'theta', 'alpha', 'beta', 'gamma']
    fooof_param_dfs = get_fooof_bp_dfs(all_fooof_parameterized_spectra, max_n_peaks=7, aperiodic_mode=param_spectra_params['aperiodic_mode'], channels=eeg_common_channels)


    out_dict = {
        'param_dfs': param_dfs,
        'fooof_param_dfs': fooof_param_dfs,
        'all_parameterized_spectra': all_parameterized_spectra,
        'all_fooof_parameterized_spectra': all_fooof_parameterized_spectra,
        'psd_dict': psd_dict,
        'param_spectra_params': param_spectra_params,
        'psd_normalize_kwargs': psd_normalize_kwargs,
        'paramtime': paramtime,
        'foooftime': foooftime,
        'psdtime': psdtime,
        'data_params': data_params,
        'epochs_params': epochs_params,
        'normalization_kwargs': normalization_kwargs,
    }
    return out_dict