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


base_dir = "output/WholeHeartData/ct/compiled-figures/dice-udf"

direct_flow_df = pd.read_csv(os.path.join(base_dir, "direct-flow.csv"))
lt_df = pd.read_csv(os.path.join(base_dir, "linear-transform.csv"))
lt_flow_df = pd.read_csv(os.path.join(base_dir, "LT-flow-udf.csv"))
# enc_lt_flow_df = pd.read_csv(os.path.join(base_dir, "encoder-LT-flow.csv"))

start = 1
step = 5
positions = [start + k * step for k in range(7)]

fig, ax = plt.subplots(figsize=(15, 5))
ax.set_ylim([0, 1])
ax.set_xlim([0, 36])
ax.plot([0., 36], 2 * [0.9], "--", color="black")

flow_bp = add_dataset(ax, direct_flow_df, start=1, step=5, color="salmon")
lt_bp = add_dataset(ax, lt_df, start=2, step=5, color="goldenrod")
lt_flow_bp = add_dataset(ax, lt_flow_df, start=3, step=5, color="olive")
# enc_lt_flow_bp = add_dataset(ax, enc_lt_flow_df, start=4, step=5, color="mediumaquamarine")
ax.legend(
    [
        flow_bp["boxes"][0],
        lt_bp["boxes"][0],
        lt_flow_bp["boxes"][0],
        # enc_lt_flow_bp["boxes"][0]
    ],
    [
        "Fl",
        "LT",
        "LT-Fl",
        "Enc-LT-Fl"
    ],
    loc="lower center"
)


xtick_positions = [2.5 + k * step for k in range(7)]
ax.set_xticks(xtick_positions, ["Myocardium", "Left Ventricle", "Right Ventricle", "Left Atrium", "Right Atrium", "Aorta", "Pulmonary Artery"])
ax.set_ylabel("Test Dice Score")
fig.tight_layout()
# fig.savefig(os.path.join(base_dir, "dice_comparison.pdf"))
# fig.savefig(os.path.join(base_dir, "dice_comparison.png"))