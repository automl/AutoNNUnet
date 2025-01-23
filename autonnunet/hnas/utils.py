import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  


def plot_norm_dist(x: np.ndarray, pdf: np.ndarray, bucket_probs: np.ndarray, bucket_centers: np.ndarray, default: int, lower: int, upper: int, title: str) -> None:
        palette = sns.color_palette("colorblind")

        plt.figure(figsize=(4 / 3 * len(bucket_probs), 4))

        plt.fill_between(x, pdf, color=palette[7], alpha=0.5)

        for i, center in enumerate(bucket_centers):
            color = palette[1] if center == default else palette[0]
            plt.bar(center, bucket_probs[i], width=0.75 , color=color, alpha=0.8)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)     
        ax.spines['right'].set_visible(False) 
        ax.spines['bottom'].set_linewidth(4)
        ax.spines['left'].set_linewidth(4)

        ax.set_xticks(bucket_centers)
        ax.set_xticklabels([""] * len(ax.get_xticks()))
        ax.tick_params(axis='x', width=4, length=10)
        ax.set_xlabel("")
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlim(lower - 1, upper + 1)

        plt.savefig(
              f"./thesis/plots/approach/hnas_distributions/{title}.svg", 
              format="svg",
              transparent=True
            )
        plt.close()


def plot_cat_dist(prior: np.ndarray, default: int, title: str) -> None:
        palette = sns.color_palette("colorblind")

        plt.figure(figsize=(4 / 3 * len(prior), 4))

        centers = np.arange(len(prior))
        for i, center in enumerate(centers):
            color = palette[1] if i == default else palette[0]
            plt.bar(center + 1, prior[i], width=0.75 , color=color, alpha=0.8)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)     
        ax.spines['right'].set_visible(False) 
        ax.spines['bottom'].set_linewidth(4)
        ax.spines['left'].set_linewidth(4)

        ax.set_xticks(centers + 1)
        ax.set_xticklabels([""] * len(ax.get_xticks()))
        ax.tick_params(axis='x', width=4, length=10)
        ax.set_xlabel("")
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlim(0, len(prior) + 1)

        plt.tight_layout()
        plt.savefig(
              f"./thesis/plots/approach/hnas_distributions/{title}.svg", 
              format="svg",
              transparent=True
            )
        plt.close()