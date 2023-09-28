import matplotlib.pyplot as plt
import pandas as pd
import os

root_dir = "/Users/arjun/Documents/Research/SimCardio/HeartFlow/output/WholeHeartData/ct/compiled-figures/sif"
df_c_05 = pd.read_csv(os.path.join(root_dir, "SIF-clip-05.csv"))
df_c_015 = pd.read_csv(os.path.join(root_dir, "SIF-clip-015.csv"))
df_div = pd.read_csv(os.path.join(root_dir, "SIF-div.csv"))

cardiac_structures = ["Myo", "LV", "RV", "LA", "RA", "Ao", "PA"]
sif_headers = [c + "-sif-percent" for c in cardiac_structures]

avg_sif_05 = [df_c_05[h].max() for h in sif_headers]
avg_sif_015 = [df_c_015[h].max() for h in sif_headers]
avg_sif_div = [df_div[h].max() for h in sif_headers]

fig, ax = plt.subplots(figsize=(15,5))
ax.set_yscale("log")
offset = 4
ax.bar(
    [offset * pos for pos, _ in enumerate(cardiac_structures)],
    avg_sif_05,
    label="clip = 0.05",
    color="salmon"
)
ax.bar(
    [offset * pos + 1 for pos, _ in enumerate(cardiac_structures)],
    avg_sif_015,
    label="clip = 0.015",
    color="goldenrod"
)
ax.bar(
    [offset * pos + 2 for pos, _ in enumerate(cardiac_structures)],
    avg_sif_div,
    label="div = 0.005",
    color="cornflowerblue"
)
ax.legend()
ax.set_ylabel("Max SIF %")

xtick_positions = [offset * k + 1 for k in range(7)]
xtick_labels = ["Myocardium", "Left Ventricle", "Right Ventricle", "Left Atrium", "Right Atrium", "Aorta", "Pulmonary Artery"]
ax.set_xticks(xtick_positions, xtick_labels)


ax.grid()

fig.savefig(os.path.join(root_dir, "max-sif-comparison.pdf"))
fig.savefig(os.path.join(root_dir, "max-sif-comparison.png"))