import matplotlib.pyplot as plt
import pandas as pd
import os

root_dir = "/Users/arjun/Documents/Research/SimCardio/HeartFlow/output/WholeHeartData/ct/compiled-figures/sif"
df_c_05 = pd.read_csv(os.path.join(root_dir, "SIF-clip-05.csv"))
df_c_015 = pd.read_csv(os.path.join(root_dir, "SIF-clip-015.csv"))

cardiac_structures = ["Myo", "LA", "LV", "RA", "RV", "Ao", "PA"]
sif_headers = [c + "-sif-percent" for c in cardiac_structures]

avg_sif_05 = [df_c_05[h].mean() for h in sif_headers]
avg_sif_015 = [df_c_015[h].mean() for h in sif_headers]
fig, ax = plt.subplots()
ax.set_yscale("log")
ax.bar(
    [3*pos for pos, _ in enumerate(cardiac_structures)],
    avg_sif_05,
    label="clip = 0.05",
    color="salmon"
)
ax.bar(
    [3*pos + 1 for pos, _ in enumerate(cardiac_structures)],
    avg_sif_015,
    label="clip = 0.015",
    color="goldenrod"
)
ax.legend()