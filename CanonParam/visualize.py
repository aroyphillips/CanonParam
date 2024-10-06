import seaborn as sns
import matplotlib.pyplot as plt
def add_significance_bar(ax, p_value, y_pos, height=0.05, fontsize=20):
    if p_value < 0.05:
        print(f"Adding significance bar at {y_pos} for p-value {p_value}")
        ax.plot([0.4, 0.6], [y_pos, y_pos], color='black', lw=2)
        ax.text(0.5, y_pos + height, "*", ha='center', va='bottom', color='black', fontsize=fontsize)


def plot_top_feats(df, y, p_values, n_feats=3, figsize=(28, 9)):
    """
    df: DataFrame with index as subject ID
    y: numpy array of binary labels
    pvals: DataFrame with p_values and p_adjusted columns
    """


    # Get the top 3 features
    top_n_features = p_values.sort_values('p_adjusted').index[:n_feats]
    top_n_feature_df = df[top_n_features].copy(deep=True)
    top_n_feature_df.loc[:, 'Label'] = y
    # Create a figure and axes
    fig, axs = plt.subplots(1, n_feats, figsize=(28, 9))

    # For each of the top 3 features
    for i, feature in enumerate(top_n_feature_df.columns):
        if feature == 'Label':
            continue
        # Create a scatterplot of the feature
        # Create a boxplot of the feature
        sns.boxplot(data=top_n_feature_df, x='Label', y=feature, ax=axs[i], hue='Label')
        # make the ylabel font bigger
        ylabel = 'Open - Closed Feature Difference'
        axs[i].set_ylabel(ylabel, fontsize=20)
        # draw a bracket with a star on the top of the plot
        p_value = p_values.loc[feature, 'p_adjusted']
        add_significance_bar(axs[i], p_value, top_n_feature_df[feature].max(), height=-.02)

        # set the title to the feature 
        feature_name = feature.replace('diff', ' Eyes Open - Closed').replace('_',' ').replace('chi', 'High Gamma')
        axs[i].set_title(feature_name.replace(' Eyes Open - Closed ', ''), fontsize=20)
        # Hide the x label
        axs[i].set_xlabel('')
        # axs[i].set_ylabel(feature.replace('diff', ' Eyes Open - Closed').replace('_',' ').replace('chi', 'High Gamma'), fontsize=20)
        # replace the xticklabels 0 and 1 with Controls and mTBI
        axs[i].set_xticks([.25, .75])
        axs[i].set_xticklabels(['Controls', 'mTBI'], fontsize=20)

    plt.show()

    return fig