## given the open_closed_pathdict from load_open_closed_dict, perform epoching
import mne
import numpy as np
import os 
import time
from joblib import Parallel, delayed

import CanonParam._temporary_data_loading.load_dataset as ld
import CanonParam._temporary_data_loading.load_open_closed_data as locd
import CanonParam._temporary_data_loading.data_utils as du


def raw2epochs(subj, subjpaths, savepath=None, window_len=10, overlap=1, include_ecg=False, channels=None, save=False, sdx=None, n_subjs=None, which_segment='both', states = ['open', 'closed']):
    """ Given a subject and a dictionary of open/closed paths, create epochs and save them
    Inputs:
        - subj (str): the subject name
        - subjpaths (dict): a dictionary of the form {'open': path, 'closed': path}
        - savepath (str): the path to save the epochs
        - window_len (int): the length of the window in seconds
        - overlap (int): the overlap of the windows in seconds
        - include_ecg (bool): whether to include the ecg channels
        - save (bool): whether to save the epochs
    Outputs:
        - dictionary of the form {'open': [open_epochs], 'closed': [closed_epochs]}
    """
    print(f'Creating epochs for {subj} {sdx+1}/{n_subjs}')
    st = time.time()
    if save:
        intermediate_savepath = os.path.join(savepath, subj)
    epochs = {state: [] for state in states}
    for state in states:
        if save:
            segpath = os.path.join(intermediate_savepath, state)
            if not os.path.exists(segpath):
                os.makedirs(segpath)
        for idx, file in enumerate(subjpaths[state]):
            if which_segment == 'first':
                if idx != 0:
                    continue
            elif which_segment == 'second':
                if idx != 1:
                    continue
            elif which_segment == 'both':
                pass
            raw = mne.io.read_raw_fif(file, preload=True, verbose=0)
            if not include_ecg:
                eeg_channels = [ch for ch in raw.ch_names if 'ECG' not in ch and 'EKG' not in ch and 'X' not in ch and 'Y' not in ch and 'Z' not in ch]
                raw = raw.pick(eeg_channels)
            if channels is not None:
                raw = raw.pick(channels)
                assert channels == raw.ch_names, f'Expected channels {channels}, got {raw.ch_names}'
            raw_epochs = mne.make_fixed_length_epochs(raw, duration=window_len, overlap=overlap, preload=True, verbose=0)
            if save:
                final_savepath = os.path.join(segpath, f'{state}{idx}_{subj}-epo.fif')
                raw_epochs.save(final_savepath)
                epochs[state].append(final_savepath)
            else:
                epochs[state].append(raw_epochs)
    print(f'Finished epochs for {subj} in {time.time()-st} seconds')
    return epochs
            
            

def create_open_closed_dict(locd_params, savepath='/shared/roy/mTBI/data_transforms/epochs/', window_len=10, overlap=9, n_jobs=1, save=False, which_split='train', tables_folder='data/tables/', max_num_subjs=None, include_ecg=False, verbosity=0):
    """ Given a params to make open/closed data, create a new pathdict with the epochs
    Inputs:
        - locd_params (dict): the parameters to pass to load_open_closed_pathdict 
        - window_len (int): the length of the window in seconds
        - overlap (int): the overlap of the windows in seconds
        - fs (int): the sampling frequency
        - channels (list): the channels to keep
        - print_statement (str): a statement to print during each epoch creation
        - verbosity (int): the verbosity level
    Outputs:
        - epoch_pathdict (dict): a dictionary of the form {subj: {'open': path, 'closed': path}}
            - each file points to the epochs numpy array of size (n_channels, n_epochs, window_length*fs)
            - the savepath for the epochs dict will store the window_len, overlap, and fs
    """
    fs = locd_params['fs_baseline']

    open_closed_pathdict = locd.load_open_closed_pathdict(**locd_params)
    epoch_dict = {}
    epochs_params = {'window_len': window_len, 'overlap': overlap, **locd_params}
    if save:
        print(f"Saving epochs to {savepath}")
        epochs_savepath = du.check_and_make_params_folder(savepath, epochs_params)
    subjs = list(open_closed_pathdict.keys())
    if which_split is not None:
        split_subjs = ld.load_splits(tables_folder=tables_folder)[which_split]
        subjs = [subj for subj in subjs if int(subj) in split_subjs]
        print(f"Using {len(subjs)} subjects for {which_split} split. {len([s for s in split_subjs if str(s) not in subjs])} subjects from the {which_split} split were not found in the open_closed_pathdict")
    if max_num_subjs is not None:
        assert type(max_num_subjs) == int, f'max_num_subjs must be an integer, not {type(max_num_subjs)}'
        subjs = subjs[:max_num_subjs]
        print(f'Using {max_num_subjs} subjects')
    processed_epochs = Parallel(n_jobs=n_jobs)(delayed(raw2epochs)(subj, open_closed_pathdict[subj], epochs_savepath, window_len=window_len, overlap=overlap, include_ecg=include_ecg, save=save, sdx=sdx, n_subjs=len(subjs)) for sdx, subj in enumerate(subjs))
    for subj, epochs in zip(subjs, processed_epochs):
        epoch_dict[subj] = epochs
    return epoch_dict

def create_random_epochs(locd_params=None, window_len=10, epochs_per_subject=10, n_jobs=1, which_segment='both', max_num_subjs=None, which_split='train', include_ecg=False, tables_folder='data/tables/', open_closed_pathdict=None):
    """ Given a params to make open/closed data, create a new dict of the form {subj: {'open': [epochs1, epochs2, ...], 'closed': [epochs1, epochs2, ...]}
    Inputs:
        - locd_params (dict): the parameters to pass to load_open_closed_pathdict
        - window_len (int): the length of the window in seconds
        - epochs_per_subject (int): the number of epochs per subject
        - which_segment (str): the segment to use, either 'first', 'second', or 'both'
        - max_num_subjs (int): the maximum number of subjects to use
        - which_split (str): the split to use (train, ival, holdout)
    Outputs:
        - epoch_dict (dict): a dictionary of the form {subj: {'open': epochs, 'closed': epochs}}
            - each file points to the epochs numpy array of size (n_channels, n_epochs, window_length*fs)
            - the savepath for the epochs dict will store the window_len, overlap, and fs
    """

    st = time.time()
    if locd_params is not None:
        print(f"Loading open/closed pathdict for random epochs")
        open_closed_pathdict = locd.load_open_closed_pathdict(tables_folder=tables_folder, **locd_params)
    elif open_closed_pathdict is None and locd_params is None:
        raise ValueError(f'Either locd_params or open_closed_pathdict must be provided')

        
    epoch_dict = {}
    subjs = list(open_closed_pathdict.keys())
    if which_split is not None:
        internal_dir = os.path.join(os.path.dirname(tables_folder[:-1]),'internal') + os.path.sep
        split_subjs = ld.load_splits(internal_folder=internal_dir)[which_split]
        subjs = [subj for subj in subjs if int(subj) in split_subjs]
        print(f"Using {len(subjs)} subjects for {which_split} split. {len([s for s in split_subjs if str(s) not in subjs])} subjects from the {which_split} split were not found in the open_closed_pathdict")
    if max_num_subjs is not None:
        assert type(max_num_subjs) == int, f'max_num_subjs must be an integer, not {type(max_num_subjs)}'
        subjs = subjs[:max_num_subjs]
        print(f'Using {max_num_subjs} subjects')
    processed_subj_paths = Parallel(n_jobs=n_jobs)(delayed(raw2randomepochs)(subj, open_closed_pathdict[subj], window_len=window_len, epochs_per_subject=epochs_per_subject, which_segment=which_segment, include_ecg=include_ecg, sdx=sdx, n_subjs=len(subjs)) for sdx, subj in enumerate(subjs))
    for subj, epochs in zip(subjs, processed_subj_paths):
        epoch_dict[subj] = epochs

    print(f'Finished random epochs for {len(epoch_dict)} subjects in {time.time()-st} seconds')
    return epoch_dict

def raw2randomepochs(subj, subjpaths, window_len=10, epochs_per_subject=10, which_segment='both', include_ecg=False, sdx=0, n_subjs=1, states = ['open', 'closed'], verbose=True):
    """ Given a subject and a dictionary of open/closed paths, create epochs_per_subject epochs of length window_len and return them
    Inputs:
        - subj (str): the subject name
        - subjpaths (dict): a dictionary of the form {'open': path, 'closed': path}
        - window_len (int): the length of the window in seconds
        - epochs_per_subject (int): the number of epochs per subject
        - which_segment (str): the segment to use, either 'first', 'second', or 'both'
    Outputs:
        - dictionary of the form {'open': [open_epochs], 'closed': [closed_epochs]} #NOTE: not preloaded
    """
    if verbose:
        print(f'Creating random epochs for {subj} {sdx+1}/{n_subjs}', end=' ... ')
    assert which_segment in ['first', 'second', 'both'], f'which_segment must be either "first", "second", or "both", not {which_segment}'
    st = time.time()
    epochs = {state: None for state in states}
    for state in states:
        temp_raw_epochs = []
        for idx, file in enumerate(subjpaths[state]):
            if which_segment == 'first':
                if idx != 0:
                    continue
            elif which_segment == 'second':
                if idx != 1:
                    continue
            elif which_segment == 'both':
                pass
            raw = mne.io.read_raw_fif(file, preload=False, verbose=0)
            if not include_ecg:
                channels = [ch for ch in raw.ch_names if 'ECG' not in ch and 'EKG' not in ch and 'X' not in ch and 'Y' not in ch and 'Z' not in ch]
                raw = raw.pick(channels)
            if max(raw.times) < window_len:
                print(f'WARNING: {subj} {state} file {file} is too short for window length {window_len}')
                continue
            raw_epochs = mne.make_fixed_length_epochs(raw, duration=window_len, overlap=window_len-0.1, preload=False, verbose=0)
            # remove annotations
            raw_epochs.annotations.delete(np.arange(len(raw_epochs.annotations.description)))
            temp_raw_epochs.append(raw_epochs)
        n_epochs = sum([epochs.events.shape[0] for epochs in temp_raw_epochs])
        if n_epochs < epochs_per_subject:
            print(f'WARNING: {subj} {state} has {n_epochs} epochs, less than the requested {epochs_per_subject}')
            continue
        rand_epochs_idxs = np.random.choice(np.arange(n_epochs), n_epochs-epochs_per_subject, replace=False)
        rand_epochs = mne.concatenate_epochs([epochs for epochs in temp_raw_epochs], verbose=0)
        rand_epochs = rand_epochs.drop(rand_epochs_idxs, verbose=0)
        assert rand_epochs.get_data().shape[0] == epochs_per_subject, f'Expected {epochs_per_subject} epochs, got {rand_epochs.get_data().shape[0]}'
        epochs[state] = rand_epochs
    if verbose:
        print(f' finished in {time.time()-st} seconds')
    return epochs

def _test_random_epochs():
    locd_params = {
        'l_freq': 0.3,
        'h_freq': None,
        'fs_baseline': 500,
        'order': 6,
        'notches': [60, 120, 180, 240],
        'notch_width': [2, 1, 0.5, 0.5],
        'num_subjs': 151,
        'reference_method': 'linked',
        'reference_channels': ['A1', 'A2'],
        'keep_refs': False,
        'bad_channels': ['T1', 'T2'],
        'filter_ecg': False,
        'ecg_l_freq': 8,
        'ecg_h_freq': 16,
        'ecg_thresh': 'auto',
        'ecg_method': 'correlation',
        'compute_bispectrum': False,
        'downsample_bispectrum': 1000,
        'load_from': 0,
        'load_to': 10,
        'random_load': False,
        'num_load_subjs': 151
    }
    epoch_dict = create_random_epochs(locd_params=locd_params, window_len=10, epochs_per_subject=5, n_jobs=1, which_segment='both')
    stacked_epochs = np.concatenate([np.concatenate(epoch_dict[subj]['open'], axis=0) for subj in epoch_dict], axis=0)
    print(f"Finished creating random epochs for {len(epoch_dict)} subjects")

def get_common_channels(epoch_dict):
    """ Given an epoch_dict, return the common channels
    Inputs:
        - epoch_dict (dict): a dictionary of the form {subj: {'open': [epochs1, epochs2, ...], 'closed': [epochs1, epochs2, ...]}
    Outputs:
        - common_channels (list): the common channels
    """
    common_channels = set()
    subjs = list(epoch_dict.keys())
    states = list(epoch_dict[subjs[0]].keys())
    for subj in subjs:
        for state in states:
            if epoch_dict[subj][state] is not None:
                if type(epoch_dict[subj][state]) == list:
                    for epochs in epoch_dict[subj][state]:
                        if len(common_channels) == 0:
                            common_channels = set(epochs.ch_names)
                        else:
                            common_channels = common_channels.intersection(set(epochs.ch_names))
                else:
                    if len(common_channels) == 0:
                        common_channels = set(epoch_dict[subj][state].ch_names)
                    else:
                        common_channels = common_channels.intersection(set(epoch_dict[subj][state].ch_names))
    eeg_channels = [ch for ch in common_channels if 'ECG' not in ch and 'EKG' not in ch and 'X' not in ch]
    eeg_sorted = sorted(eeg_channels)
    ecg_channels = [ch for ch in common_channels if 'ECG' in ch or 'EKG' in ch or 'X' in ch]
    if len(ecg_channels) > 0:
        ecg_sorted = sorted(ecg_channels)
        common_channels = eeg_sorted + ecg_sorted
    else:
        common_channels = eeg_sorted
    return common_channels


def get_eeg_channels(channels):
    eeg_channels = [ch for ch in channels if 'ECG' not in ch and 'EKG' not in ch and 'X' not in ch]
    return eeg_channels

def get_ecg_channels(channels):
    ecg_channels = [ch for ch in channels if 'ECG' in ch or 'EKG' in ch or 'X' in ch]
    return ecg_channels

def reorder_epoch_dict(epoch_dict):
    """ Given an epoch_dict, ensure that the order the channels are in is consistent
    Inputs:
        - epoch_dict (dict): a dictionary of the form {subj: {'open': [epochs1, epochs2, ...], 'closed': [epochs1, epochs2, ...]}
    Outputs:
        - epoch_dict (dict): a dictionary of the form {subj: {'open': [epochs1, epochs2, ...], 'closed': [epochs1, epochs2, ...]}
            - the order of the channels in each epoch will be consistent

    OPERATES IN MEMORY
    """
    subjs = list(epoch_dict.keys())
    states = list(epoch_dict[subjs[0]].keys())
    new_ecg_name = ['ECG']
    common_channels = get_common_channels(epoch_dict)
    ecg_common_channels = common_channels.copy() + new_ecg_name
    has_ecg = False
    for subj in subjs:
        for state in states:
            if epoch_dict[subj][state] is not None:
                if type(epoch_dict[subj][state]) == list:
                    temp_epochs = []
                    for epochs in epoch_dict[subj][state]:
                        ecg_inst_channel = [ch for ch in epochs.ch_names if 'ECG' in ch or 'EKG' in ch or 'X' in ch]
                        assert len(ecg_inst_channel) <= 1, f'Expected at most 1 ECG channel, got {len(ecg_inst_channel)} for subj {subj} state {state}'
                        if len(ecg_inst_channel) == 1:
                            epochs.rename_channels({ecg_inst_channel[0]: new_ecg_name[0]})
                            temp_epochs.append(epochs.pick(ecg_common_channels))
                            has_ecg = True
                        else:
                            temp_epochs.append(epochs.pick(common_channels))
                            if has_ecg:
                                raise ValueError(f'Expected ECG channel in {subj} {state} epochs')
                    epoch_dict[subj][state] = temp_epochs
                elif type(epoch_dict[subj][state]) == mne.epochs.EpochsArray: # raw would probably work just as well
                    ecg_inst_channel = [ch for ch in epoch_dict[subj][state].ch_names if 'ECG' in ch or 'EKG' in ch or 'X' in ch]
                    assert len(ecg_inst_channel) <= 1, f'Expected at most 1 ECG channel, got {len(ecg_inst_channel)} for subj {subj} state {state}'
                    if len(ecg_inst_channel) == 1:
                        epoch_dict[subj][state].rename_channels({ecg_inst_channel[0]: new_ecg_name[0]})
                        epoch_dict[subj][state] = epoch_dict[subj][state].pick(ecg_common_channels)
                        has_ecg = True
                    else:
                        epoch_dict[subj][state] = epoch_dict[subj][state].pick(common_channels)
                        if has_ecg:
                            raise ValueError(f'Expected ECG channel in {subj} {state} epochs')
                else:
                    raise ValueError(f'Expected epochs to be either a list or an EpochsArray, not {type(epoch_dict[subj][state])}')
    return epoch_dict

def main_make_epochs(locd_params, epoch_method='random', window_len=10, overlap=9, epochs_per_subject=10, which_segment='both', savepath='/shared/roy/mTBI/data_transforms/epochs/', n_jobs=1, save=False, tables_folder='data/tables/', max_num_subjs=None, which_split='train', verbosity=0):
    """ Given the parameters to make epochs, create the epochs and save them
    Inputs:
        - locd_params (dict): the parameters to pass to load_open_closed_pathdict
        - epoch_method (str): the method to use to create the epochs, either 'random' or 'fixed'
        - window_len (int): the length of the window in seconds
        - overlap (int): the overlap of the windows in seconds
        - epochs_per_subject (int): the number of epochs per subject
        - which_segment (str): the segment to use, either 'first', 'second', or 'both'
        - savepath (str): the path to save the epochs
        - n_jobs (int): the number of jobs to use
        - save (bool): whether to save the epochs
        - verbosity (int): the verbosity level
    Outputs:
        - epoch_dict (dict): a dictionary of the form {subj: {'open': [epochs1, epochs2, ...], 'closed': [epochs1, epochs2, ...]} if save=False
            - if save=true each file points to the epochs numpy array of size (n_channels, n_epochs, window_length*fs)
            - the savepath for the epochs dict will store the window_len, overlap, and fs
    """
    include_ecg = False
    if 'include_ecg' in locd_params.keys():
        if locd_params['include_ecg']:
            include_ecg = True
    if epoch_method == 'fixed':
        epoch_dict = create_open_closed_dict(locd_params, savepath=savepath, window_len=window_len, overlap=overlap, n_jobs=n_jobs, verbosity=verbosity, save=save, which_split=which_split, tables_folder=tables_folder, max_num_subjs=max_num_subjs, include_ecg=include_ecg)
    elif epoch_method == 'random':
        epoch_dict = create_random_epochs(locd_params=locd_params, window_len=window_len, epochs_per_subject=epochs_per_subject, n_jobs=n_jobs, which_segment=which_segment, max_num_subjs=max_num_subjs, which_split=which_split, include_ecg=include_ecg, tables_folder=tables_folder)
    else:
        raise ValueError(f'epoch_method must be either "fixed" or "random", not {epoch_method}')
    if not save:
        epoch_dict = reorder_epoch_dict(epoch_dict)
    return epoch_dict


if __name__ == '__main__':
    _test_random_epochs()
    print('Done')
