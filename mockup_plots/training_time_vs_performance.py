import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


PLOTS_DIR = Path("./mockup_plots").resolve()


def plot(step: int, prev_g=None):
    # Data from https://arxiv.org/pdf/2404.09556
    data = pd.DataFrame({
        "Model": ["nnU-Net (org.)", "nnU-Net ResEnc M", "nnU-Net ResEnc L", "nnU-Net ResEnc XL", "MedNeXt L k3", "MedNeXt L k5"],
        "Type": ["nnU-Net", "nnU-Net ResEnc", "nnU-Net ResEnc", "nnU-Net ResEnc", "MedNeXt", "MedNeXt"],
        "Runtime [h]": [9, 12, 35, 66, 68, 233],
        "Dice": [
            [83.08, 91.54, 80.09, 91.24, 86.04, 88.64],
            [83.31, 91.99, 80.75, 91.26, 86.79, 88.77],
            [83.35, 91.69, 81.60, 91.13, 88.17, 89.41],
            [83.28, 91.48, 81.19, 91.18, 88.67, 89.68],
            [84.70, 92.65, 82.14, 91.35, 88.25, 89.62],
            [85.04, 92.62, 82.34, 91.50, 87.74, 89.73]        
        ]
    })

    types = list(data["Type"].unique())[:step + 1]
    data = data[data["Type"].isin(types)]

    data["Dice"] = data["Dice"].apply(lambda x: np.mean(x))

    fig = plt.figure(figsize=(10, 3))
    g = sns.scatterplot(data=data, y="Runtime [h]", x="Dice", s=100, hue="Type", palette="colorblind")
    type_colors = {
        "nnU-Net": sns.color_palette("colorblind")[0],
        "nnU-Net ResEnc": sns.color_palette("colorblind")[1],
        "MedNeXt": sns.color_palette("colorblind")[2],
    }

    for i, row in data.iterrows():
        color = type_colors[row["Type"]]
        plt.plot([row["Dice"], row["Dice"]], [0, row["Runtime [h]"]], color=color, linewidth=3)

    g.set_xlim(86.4275, 88.4825)
    g.set_ylim(0, 250)

    g.set_xlabel("Dice Score [%]")
    g.set_ylabel("Training Time [h]")

    if prev_g:
        handles, labels = prev_g.get_legend_handles_labels()
    else:
        handles, labels = None, None

    g.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        fancybox=False,
        shadow=False,
        frameon=False,
    )

    fig.subplots_adjust(
        top=0.96,   
        bottom=0.3, 
        left=0.08,  
        right=0.99,  
        # wspace=0.1  
    )


    plt.grid(True)

    plt.savefig(PLOTS_DIR / f"training_time_vs_performance_{step}.png", dpi=400)

    if prev_g:
        return prev_g
    else:
        return g

if __name__ == "__main__":
    sns.set_theme(style="darkgrid")
    sns.set_palette("colorblind")

    g = None
    for step in reversed(range(3)):
        g = plot(step, prev_g=g)