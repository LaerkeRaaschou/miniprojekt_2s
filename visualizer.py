import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class FeatureVisualizer:
    @staticmethod
    def apply_pca(data):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        pca = PCA(n_components=3)
        return pca.fit_transform(scaled)

    @staticmethod
    def plot_3d(dataset, target):
        color_map = {
            "grassland": "cyan", "lake": "blue", "forest": "green", "swamp": "brown",
            "field": "yellow", "mine": "purple", "home": "orange", "table": "pink", "wrong": "red"
        }
        colors = np.array([color_map[label] for label in target])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
        ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=colors, s=40)
        ax.set(title="Plot in 3 Dimensions", xlabel="1st Eigenvector", ylabel="2nd Eigenvector", zlabel="3rd Eigenvector")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        legend_labels = [
            plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor=color_map[key], label=key)
            for key in color_map
        ]
        ax.legend(handles=legend_labels, title="Classes", loc="upper right")
        plt.show()