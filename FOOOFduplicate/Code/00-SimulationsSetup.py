# %matplotlib inline
from fooof import FOOOF
from fooof.sim import gen_power_spectrum
from fooof.plts import plot_spectrum

# Import project specific (local) custom code
import sys
sys.path.append('../code')
from settings import *
from utils import print_settings

# Simulation settings
ap = [0, 1]
osc = [10, 0.25, 1.5]
nlv = 0.0

# Generate the spectrum
freqs, powers = gen_power_spectrum(F_RANGE, ap, osc, nlv, F_RES)

# Plot example simulated power spectrum
plot_spectrum(freqs, powers, log_powers=True)


# Parameterize our simulated spectrum
fm = FOOOF(aperiodic_mode='fixed', verbose=False)
fm.report(freqs, powers, plt_log=False)

# Compare actual parameters to FOOOF fit
print(ap, '\t', fm.aperiodic_params_)
print(osc, '\t', fm.gaussian_params_)


# Settings
ap = [1, 100, 2]
osc = [10, 0.3, 1]
nlv = 0.

# Simulate the power spectrum
fs, ps = gen_power_spectrum(F_RANGE_LONG, ap, osc, nlv, F_RES_LONG)

# Parameterize the power spectrum
fm = FOOOF(aperiodic_mode='knee', verbose=False)
fm.report(fs, ps, plt_log=True)

# Compare actual parameters to FOOOF fit
print(ap, '\t', fm.aperiodic_params_)
print(osc, '\t', fm.gaussian_params_)

# Check the settings for the number of oscillations
print_settings(N_PEAK_OPTS, N_PEAK_PROBS, '# of oscs')

# Check the settings for the oscillation center frequencies
print_settings(CF_OPTS, CF_PROBS, 'CFs')

# Check the settings for the oscillation powers
print_settings(PW_OPTS, PW_PROBS, 'PWs')

# Check the settings for the oscillation bandwidths
print_settings(BW_OPTS, BW_PROBS, 'BWs')

# Check the settings for the aperiodic offsets
print_settings(OFF_OPTS, OFF_PROBS, 'aperiodic offset')

# Check the settings for the aperiodic knee
print_settings(KNE_OPTS, KNE_PROBS, 'aperiodic knee')

# Check the settings for the aperiodic exponents
print_settings(EXP_OPTS, EXP_PROBS, 'aperiodic exponent')


# Sanity check probabilities are set correctly
for probs in [CF_PROBS, PW_PROBS, BW_PROBS, OFF_PROBS, KNE_PROBS, EXP_PROBS]:
    assert np.isclose(sum(probs), 1, 1e-10)
print('All probabilities check out.')


# Check the settings used for fitting the smaller frequency range
FOOOF(*FOOOF_SETTINGS).print_settings()


# Check the settings used for fitting across the larger frequency range
FOOOF(*FOOOF_SETTINGS_KNEE).print_settings()

# Set whether to check and build folders
BUILD_FOLDERS = True

# Check and build folders
if BUILD_FOLDERS:

    import os
    from settings import FOLDER_NAMES

    for folder in ['../data', '../figures']:
        if not os.path.exists(folder):
            os.mkdir(folder)
            for sub_folder in FOLDER_NAMES:
                os.mkdir(os.path.join(folder, sub_folder))