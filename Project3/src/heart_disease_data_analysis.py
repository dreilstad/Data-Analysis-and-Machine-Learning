import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# reads the data
df = pd.read_csv("data/heart_disease.csv")

# cleans up data
features = df.columns.to_numpy()[:-1]
data = df.iloc[:, :-1]
targets = df["target"].to_numpy()
targets = np.where(targets > 0, 1, targets)


def correlation_matrix():
    """
    Function plots the correlation matrix with the features in the dataset.
    """

    correlation_matrix = pd.DataFrame(data, columns=features).corr().round(1)
    mask = np.zeros_like(correlation_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(15,8))
    ax = sns.heatmap(data=correlation_matrix, mask=mask, annot=True, annot_kws={"size":16})
    sns.set(font_scale=1.4)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 16)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 16)
    plt.title("Heart Disease - Correlation matrix")
    plt.tight_layout()
    plt.savefig("heart_disease_corr_matrix.png", dpi=300)
    plt.show()

def principal_component_analysis():
    """
    Function performs a principal component analysis of the data, 
    and plots the explained variance ratios for each feature as a bar plot.
    """

    X = data.to_numpy()

    pca = PCA()
    pca = pca.fit(X)

    print("PCA explained variance ratios: \n" + str(pca.explained_variance_ratio_))
    print("\nPCA sum of all variance ratios: " + str(np.sum(pca.explained_variance_ratio_)))
    print("\nPCA sum of first 3 variance ratios: " + str(np.sum(pca.explained_variance_ratio_[:4])))

    plt.bar(features, pca.explained_variance_ratio_)
    plt.xticks(ticks=features, labels=features, rotation=90)
    plt.yticks(ticks=np.arange(0.0, 1.1, 0.1))
    plt.title("Principal Component Analysis - explained variance ratios")
    plt.tight_layout()
    plt.savefig("pca_variance_ratio.png", dpi=300)
    plt.show()

def feature_histogram():
    """
    Function plots the histograms for each class for each feature in the dataset
    """

    fig, axes = plt.subplots(7,2,figsize=(15,15))
    presence0 = data[targets == 0]
    presence1 = data[targets == 1]
    ax = axes.ravel()

    for i in range(13):
        _, bins = np.histogram(data.iloc[:,i], bins=25)
        ax[i].hist(presence0.iloc[:,i], bins = bins, alpha = 0.5)
        ax[i].hist(presence1.iloc[:,i], bins = bins, alpha = 0.5)
        ax[i].set_title(features[i])
        ax[i].set_yticks(())

    ax[0].set_xlabel("Feature magnitude")
    ax[0].set_ylabel("Frequency")
    ax[0].legend(["Presence = 0", "Presence = 1"], loc ="best")
    fig.tight_layout()
    plt.savefig("heart_disease_feature_histograms.png", dpi=300)
    plt.show()

if __name__ == "__main__":

    method = sys.argv[1]

    if method == "correlation":
        correlation_matrix()
    elif method == "pca":
        principal_component_analysis()
    elif method == "histogram":
        feature_histogram()
