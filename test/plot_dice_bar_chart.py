import pandas as pd
import matplotlib.pyplot as plt

# plotting blank plot with 0.9 as benchmark score
fig, ax = plt.subplots()
ax.plot([0., 9.], 2 * [0.9], "--", color="black")
ax.set_ylim([0., 1.])
ax.grid()
ax.set_xlim([0., 9.])
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], ["Myo", "LV", "RV", "LA", "RA", "Ao", "PA", "WH"])
ax.set_ylabel("Dice")
fig.tight_layout()
fig.savefig("output/misc/dice_baseline.png")

csv_file = "output/WholeHeart/ct/combined-4/evaluation/test/ct/ct_compiled.csv"
df = pd.read_csv(csv_file)
fig, ax = plt.subplots()
ax.plot([0., 9.], 2 * [0.9], "--", color="black")
ax.set_ylim([0., 1.])
ax.set_xlim([0., 9.])
ax.grid()
ax.boxplot(df[["Epi", "LV", "RV", "LA", "RA", "Ao", "PA", "WH"]], patch_artist=True,
           boxprops=dict(facecolor="blue", alpha=0.5))
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], ["Myo", "LV", "RV", "LA", "RA", "Ao", "PA", "WH"])
ax.set_ylabel("Dice")
fig.tight_layout()
fig.savefig("output/WholeHeart/ct/combined-4/evaluation/test/ct/dice.png")

csv_file = "output/WholeHeart/ct/combined-4/evaluation/test/ct/ct_compiled.csv"
df = pd.read_csv(csv_file)




# Blank plot
# plotting blank plot with 0.9 as benchmark score
fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios":[6,1]})
ax[0].plot([0., 8.], 2 * [0.9], "--", color="black")
ax[0].set_ylim([0, 1])
ax[0].set_xlim([0,8])
ax[0].grid()
ax[0].set_ylabel("Dice")
ax[0].set_xticks([1, 2, 3, 4, 5, 6, 7], ["Myo", "LV", "RV", "LA", "RA", "Ao", "PA"])

ax[1].plot([0., 2.], 2 * [0.9], "--", color="black")
ax[1].set_xlim([0.,2.])
ax[1].set_xticks([1], ["WH"])
ax[1].grid()
fig.savefig("output/misc/dice_separate_WH_baseline.pdf")





# Plot with WH separate and colored individually
fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios":[6,1]})
ax[0].plot([0., 8.], 2 * [0.9], "--", color="black")
ax[0].set_ylim([0, 1])
ax[0].set_xlim([0,8])
ax[0].set_ylabel("Dice")
ax[0].boxplot(df["Epi"], positions=[1], widths=[0.75], patch_artist=True, boxprops=dict(facecolor="blue", alpha=1.0), medianprops = dict(color="white"))
ax[0].boxplot(df["LV"], positions=[2], widths=[0.75], patch_artist=True, boxprops=dict(facecolor="lightskyblue", alpha=1.0), medianprops = dict(color="white"))
ax[0].boxplot(df["RV"], positions=[3], widths=[0.75], patch_artist=True,
              boxprops=dict(facecolor="sandybrown", alpha=1.0), medianprops = dict(color="white"))
ax[0].boxplot(df["LA"], positions=[4], widths=[0.75], patch_artist=True,
              boxprops=dict(facecolor="cornflowerblue", alpha=1.0), medianprops = dict(color="white"))
ax[0].boxplot(df["RA"], positions=[5], widths=[0.75], patch_artist=True, boxprops=dict(facecolor="darkgray", alpha=1.0), medianprops = dict(color="white"))
ax[0].boxplot(df["Ao"], positions=[6], widths=[0.75], patch_artist=True, boxprops=dict(facecolor="salmon", alpha=1.0), medianprops = dict(color="white"))
ax[0].boxplot(df["PA"], positions=[7], widths=[0.75], patch_artist=True, boxprops=dict(facecolor="crimson", alpha=1.0), medianprops = dict(color="white"))
ax[0].set_xticks([1, 2, 3, 4, 5, 6, 7], ["Myo", "LV", "RV", "LA", "RA", "Ao", "PA"])
ax[0].grid()

ax[1].plot([0., 2.], 2 * [0.9], "--", color="black")
ax[1].set_xlim([0.,2.])
ax[1].boxplot(df["WH"], positions=[1], widths=[1.0], patch_artist=True, boxprops=dict(facecolor="dimgray", alpha=1.0), medianprops = dict(color="white"))
ax[1].set_xticks([1], ["WH"])
ax[1].grid()
fig.savefig("output/WholeHeart/ct/combined-4/evaluation/test/ct/dice_boxplot_separate_WH.pdf")