from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
import CanonParam._temporary_data_loading.load_dataset as ld
LABEL_DICT = ld.load_label_dict()

def get_mwup_values(df, y, adjust_method='holm'):

    """
    df: DataFrame with index as subject ID
    y: numpy array of binary labels
    adjust_method: method for multiple testing correction (default is 'holm')

    """
    assert len(df) == len(y), "Length of dataframe and labels do not match"
    p_values = []

    for col in df.columns:
        sample1 = df[col][y == 0]
        sample2 = df[col][y == 1]
        _, p_val = mannwhitneyu(sample1, sample2)
        p_values.append(p_val)
    p_adjusted = multipletests(p_values, method='holm')[1]
    p_df = pd.DataFrame({'p_values': p_values, 'p_adjusted': p_adjusted}, index=df.columns)
    return p_df


def get_y_from_df(df, label_dict=LABEL_DICT):
    """
    Returns the binary diagnosis labels from a dataframe that has the index as the subject ID
    Inputs:
        df: DataFrame with index as subject ID
        label_dict: dictionary mapping subject ID to diagnosis (default is LABEL_DICT)
    Outputs:
        y: numpy array of binary labels
    """
    assert all([int(ind) in label_dict for ind in df.index]), "Not all indices in the dataframe are in the label dictionary"
    return np.array([label_dict[int(ind)] for ind in df.index])