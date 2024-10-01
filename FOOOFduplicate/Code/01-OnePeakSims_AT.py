from os.path import join as pjoin

import numpy as np
from scipy.stats import spearmanr

from fooof import FOOOFGroup, fit_fooof_3d
from fooof.sim import gen_group_power_spectra
from fooof.sim.utils import set_random_seed

# Import project specific (local) custom code
import sys
sys.path.append('../code')
from plts import *
from sims import *
from utils import *
from analysis import *
from settings import *

print('Imports done!')


# Set random seed
set_random_seed(303)

# Set a folder name (for saving data & figures)
FOLDER = '01_one-peak/'

# Data & Model Setting
GEN_SIMS = True
SAVE_SIMS = True
FIT_MODELS = True
SAVE_MODELS = True

# Run Settings
SAVE_FIG = False
PLT_LOG = True

# If not plotting in log, turn off log-defined plot limits
if not PLT_LOG:
    YLIMS_AP = YLIMS_KN = YLIMS_CF = YLIMS_PW = YLIMS_BW = None
    YTICKS_CF = YTICKS_PW = YTICKS_BW = None

# Check the conditions to simulate across: noise level
print('NLVS: ', NLVS)

# Set the number of power spectra (per condition)
n_psds = N_PSDS

# Use generators to sample peak & aperiodic parameters
peaks = gen_peak_def(1)
aps = gen_ap_def()

# Get data sizes
n_conds = len(NLVS)
n_freqs = int((F_RANGE[1] - F_RANGE[0]) / F_RES + 1)

# Generate or load power spectra
data_name = 'single_peak_sims'

if GEN_SIMS:
    
    # Initialize data stores
    psds = np.empty(shape=[n_conds, n_psds, n_freqs])
    sim_params = [None] * n_conds

    # Generate simulated power spectra
    for n_ind, nlv in enumerate(NLVS):
        freqs, psds[n_ind, :, :], sim_params[n_ind] = \
            gen_group_power_spectra(n_psds, F_RANGE, aps, peaks, nlv,
                                    F_RES, return_params=True)
    
    # Save out generated simulated data & parameter definitions
    if SAVE_SIMS:
        save_sim_data(data_name, FOLDER, freqs, psds, sim_params)

else:
    # Reload simulated data and parameter definitions
    freqs, psds, sim_params = load_sim_data(data_name, FOLDER)
    
# Check shape
print('n_conds, n_spectra, n_freqs : ', psds.shape)

# Extract ground truth values
peak_truths, ap_truths = get_ground_truth(sim_params)

emp_nlv = []
for nlv in NLVS:
    temp = []

    # Within each noise level, get the average squared 'error'
    for it in range(n_psds):
        temp.append(np.mean(np.random.normal(0, nlv, len(freqs))**2))
    
    # Get the average 'error' per noise level
    emp_nlv.append(np.mean(temp))


# Compare simulated values to empirical values
print('Simulated NLV Values:')
print_list(NLVS)
print('Empirical NLV Values:')
print_list(np.sqrt(emp_nlv))

## Fit Power spectra with FOOOF
# Initialize FOOOFGroup to test with
fg = FOOOFGroup(*FOOOF_SETTINGS, verbose=False)

# Print out settings used for fitting simulated power spectra
fg.print_settings()

# Fit power spectra
if FIT_MODELS:
    fgs = fit_fooof_3d(fg, freqs, psds)
    
    if SAVE_MODELS:
        save_model_data(data_name, FOLDER, fgs)
            
else:
    # Reload model fit data
    fgs = load_model_data(data_name, FOLDER, n_conds)

print('fgs shape:', len(fgs))
print('fiting done!')

# # Alternatively:
# # Fit power spectra - adapting FOOOF settings for each group of simulated power spectra
# fgs = []
# for nlv, nlv_psds in zip(nlvs, psds):
#     fg = FOOOFGroup(peak_width_limits=[1, 6], max_n_peaks=6, min_peak_height=5*nlv)
#     fg.fit(freqs, nlv_psds)
#     fgs.append(fg.copy())

# Extract data of interest from FOOOF fits
peak_fits, ap_fits, err_fits, r2_fits, n_fit_peaks = get_fit_data(fgs)


# Calculate errors
peak_errors = calc_errors(peak_truths, peak_fits)
ap_errors = calc_errors(ap_truths, ap_fits)


# Pull out error per peak parameter
cf_errors = peak_errors[:, :, 0]
pw_errors = peak_errors[:, :, 1]
bw_errors = peak_errors[:, :, 2]

# Pull out error per aperiodic parameter
off_errors = ap_errors[:, :, 0]
exp_errors = ap_errors[:, :, 1]

# Check overall fit quality
temp = r2_fits.flatten()
print('Min/Max R^2: \t{:1.4f} \t{:1.4f}'.format(np.nanmin(temp), np.nanmax(temp)))
print('Median  R^2: \t{:1.4f}'.format(np.nanmedian(temp)))

# Collect data together
datas = {
    'CF' : cf_errors,
    'PW' : pw_errors,
    'BW' : bw_errors,
    'OFF' : off_errors,
    'EXP' : exp_errors,
    'ERR' : err_fits,
    'R^2' : r2_fits
}


# Print out the average error for each parameter
#   Also prints out the average model error and R^2 per noise level
with np.printoptions(precision=4, suppress=True):
    for label, data in datas.items():
        print(label, '\n\t', np.nanmedian(data, 1))

# Check if the amount of error correlates with the noise level
print('Correlations of median error with noise level:')
print('\n\t\t  r-val\t\tp-val')
print('\tCF\t: {:1.3f} \t{:1.2f}'.format(
    *spearmanr(NLVS, np.nanmedian(cf_errors, axis=1))))
print('\tPW\t: {:1.3f} \t{:1.2f}'.format(
    *spearmanr(NLVS, np.nanmedian(pw_errors, axis=1))))
print('\tBW\t: {:1.3f} \t{:1.2f}'.format(
    *spearmanr(NLVS, np.nanmedian(bw_errors, axis=1))))
print('\tOFF\t: {:1.3f} \t{:1.2f}'.format(
    *spearmanr(NLVS, np.nanmedian(off_errors, axis=1))))
print('\tEXP\t: {:1.3f} \t{:1.2f}'.format(
    *spearmanr(NLVS, np.nanmedian(exp_errors, axis=1))))


# Plot the error of peak center frequency reconstruction
plot_errors_violin(cf_errors, 'Center Frequency', plt_log=PLT_LOG, 
                   ylim=YLIMS_CF, yticks=YTICKS_CF,
                   save_fig=SAVE_FIG, save_name=pjoin(FOLDER, 'cf_error'))


# Plot the error of peak power reconstruction
plot_errors_violin(pw_errors, 'Power', plt_log=PLT_LOG, 
                   ylim=YLIMS_PW, yticks=YTICKS_PW,
                   save_fig=SAVE_FIG, save_name=pjoin(FOLDER, 'pw_error'))


# Plot the error of peak bandwidth reconstruction
plot_errors_violin(bw_errors, 'Bandwidth', plt_log=PLT_LOG,
                   ylim=YLIMS_BW, yticks=YTICKS_BW,
                   save_fig=SAVE_FIG, save_name=pjoin(FOLDER, 'bw_error'))

# Plot the number of peaks fit per condition
n_peak_counter = count_peak_conditions(n_fit_peaks, NLVS)
plot_n_peaks_bubbles(n_peak_counter, ms_val=15, x_label='nlvs')

# Plot the error of aperiodic offset reconstruction
plot_errors_violin(off_errors, 'Offset', plt_log=PLT_LOG, ylim=YLIMS_AP,
                   save_fig=SAVE_FIG, save_name=pjoin(FOLDER, 'off_error'))

# Plot the error of aperiodic exponent reconstruction
plot_errors_violin(exp_errors, 'Exponent', plt_log=PLT_LOG, ylim=YLIMS_AP,
                   save_fig=SAVE_FIG, save_name=pjoin(FOLDER, 'exp_error'))


# Plot the amount of error across noise levels
plot_errors_violin(err_fits, 'Fit Error', y_label='Error', plt_log=False,
                   save_fig=SAVE_FIG, save_name=pjoin(FOLDER, 'model_error'))


# Plot the goodness-of-fit (R^2) across noise levels
plot_errors_violin(r2_fits, 'R2', y_label='R^2', plt_log=False,
                   save_fig=SAVE_FIG, save_name=pjoin(FOLDER, 'model_r_squared'))


### the Example FOOOF Fits (in the original .ipynb file) are not included here


