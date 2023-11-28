import pandas as pd
import matplotlib.pyplot as plt
import os


def add_dataset(ax, df, start, step, color, width=1):
    ax.boxplot(
        df["Epi"],
        positions=[start],
        widths=[width],
        patch_artist=True,
        boxprops=dict(
            facecolor=color,
            alpha=1.0
        ),
        medianprops=dict(color="white"),
    )
    ax.boxplot(
        df["LV"],
        positions=[start + step],
        widths=[width],
        patch_artist=True,
        boxprops=dict(
            facecolor=color,
            alpha=1.0
        ),
        medianprops=dict(color="white")
    )
    ax.boxplot(
        df["RV"],
        positions=[start + 2 * step],
        widths=[width],
        patch_artist=True,
        boxprops=dict(
            facecolor=color,
            alpha=1.0
        ),
        medianprops=dict(color="white")
    )
    ax.boxplot(
        df["LA"],
        positions=[start + 3 * step],
        widths=[width],
        patch_artist=True,
        boxprops=dict(
            facecolor=color,
            alpha=1.0
        ),
        medianprops=dict(color="white")
    )
    ax.boxplot(
        df["RA"],
        positions=[start + 4 * step],
        widths=[width],
        patch_artist=True,
        boxprops=dict(
            facecolor=color,
            alpha=1.0
        ),
        medianprops=dict(color="white")
    )
    ax.boxplot(
        df["Ao"],
        positions=[start + 5 * step],
        widths=[width],
        patch_artist=True,
        boxprops=dict(
            facecolor=color,
            alpha=1.0
        ),
        medianprops=dict(color="white")
    )
    bp = ax.boxplot(
        df["PA"],
        positions=[start + 6 * step],
        widths=[width],
        patch_artist=True,
        boxprops=dict(
            facecolor=color,
            alpha=1.0
        ),
        medianprops=dict(color="white")
    )

    return bp


base_dir = "output/WholeHeartData/ct-mr-cropped/compiled-figures/dice-ablation/mr"
fontsize = 24

fl_df = pd.read_csv(os.path.join(base_dir, "FL.csv"))
lt_df = pd.read_csv(os.path.join(base_dir, "LT.csv"))
lt_fl_df = pd.read_csv(os.path.join(base_dir, "LT-FL.csv"))
lt_fl_v_df = pd.read_csv(os.path.join(base_dir, "LT-FL-v.csv"))

start = 1
step = 5
positions = [start + k * step for k in range(7)]

fig, ax = plt.subplots(figsize=(15, 5))
ax.set_ylim([0, 1])
ax.set_xlim([0, 35])
ax.plot([0., 36], 2 * [0.9], "--", color="black")

flow_bp = add_dataset(ax, fl_df, start=1, step=step, color="salmon")
lt_bp = add_dataset(ax, lt_df, start=2, step=step, color="olive")
lt_fl_bp = add_dataset(ax, lt_fl_df, start=3, step=step, color="goldenrod")
lt_fl_v_bp = add_dataset(ax, lt_fl_v_df, start=4, step=step, color="cornflowerblue")
ax.legend(
    [
        flow_bp["boxes"][0],
        lt_bp["boxes"][0],
        lt_fl_bp["boxes"][0],
        lt_fl_v_bp["boxes"][0]
    ],
    [
        "FL",
        "LT",
        "LT-FL",
        "LT-FL-v"
    ],
    loc="lower center",
    fontsize=fontsize
)


xtick_positions = [2.5 + k * step for k in range(7)]
ax.tick_params(axis="both", labelsize=fontsize)
ax.set_xticks(xtick_positions, ["Epi", "LA", "LV", "RA", "RV", "Ao", "PA"])
ax.set_ylabel("Test Dice Score", fontsize=fontsize)
fig.tight_layout()
fig.savefig(os.path.join(base_dir, "dice-comparison.pdf"))
fig.savefig(os.path.join(base_dir, "dice-comparison.png"))