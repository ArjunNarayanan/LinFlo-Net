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

# direct_flow_df = pd.read_csv(os.path.join(base_dir, "direct-flow.csv"))
# lt_df = pd.read_csv(os.path.join(base_dir, "linear-transform.csv"))
lt_flow_occupancy = pd.read_csv(os.path.join(base_dir, "LT-flow-occupancy.csv"))
lt_flow_udf = pd.read_csv(os.path.join(base_dir, "LT-flow-udf.csv"))

start = 1
step = 4
positions = [start + k * step for k in range(7)]

fig, ax = plt.subplots(figsize=(15, 5))
ax.set_ylim([0, 1])
ax.set_xlim([0, 28])
ax.plot([0., 36], 2 * [0.9], "--", color="black")

# flow_bp = add_dataset(ax, direct_flow_df, start=1, step=5, color="salmon")
lt_flow_occ_bp = add_dataset(ax, lt_flow_occupancy, start=1, step=step, color="olive")
lt_flow_udf_bp = add_dataset(ax, lt_flow_udf, start=2, step=step, color="goldenrod")
ax.legend(
    [
        # flow_bp["boxes"][0],
        lt_flow_occ_bp["boxes"][0],
        # lt_flow_bp["boxes"][0],
        lt_flow_udf_bp["boxes"][0]
    ],
    [
        "Occ",
        "UDF",
    ],
    loc="lower center"
)


xtick_positions = [1.5 + k * step for k in range(7)]
ax.set_xticks(xtick_positions, ["Myocardium", "Left Ventricle", "Right Ventricle", "Left Atrium", "Right Atrium", "Aorta", "Pulmonary Artery"])
ax.set_ylabel("Test Dice Score")
fig.tight_layout()
# fig.savefig(os.path.join(base_dir, "dice_comparison.pdf"))
fig.savefig(os.path.join(base_dir, "occ-vs-udf.png"))