import pandas as pd
import matplotlib.pyplot as plt




csv_file = "output/WholeHeartData/trained_models/ct/flow/model-1/evaluation/test/ct/ct_compiled.csv"
df = pd.read_csv(csv_file)

fig, ax = plt.subplots()
ax.set_ylim([0.,1.])
ax.grid()
ax.boxplot(df[["Epi", "LV", "RV", "LA", "RA", "Ao", "PA", "WH"]], patch_artist=True, boxprops=dict(facecolor="blue", alpha=0.5))
ax.set_xticks([1,2,3,4,5,6,7,8], ["Myo", "LV", "RV", "LA", "RA", "Ao", "PA", "WH"])
ax.set_ylabel("Dice")
fig.tight_layout()
fig.savefig("output/WholeHeartData/trained_models/ct/flow/model-1/evaluation/test/ct/dice.png")