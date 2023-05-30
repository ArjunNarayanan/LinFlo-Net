import pandas as pd
import os
import numpy as np

input_folder = "output/WholeHeart/ct/combined-4/evaluation/test/ct/dice"
output_folder = "output/WholeHeart/ct/combined-4/evaluation/test/ct/"

modality = "ct"
all_scores = []
all_samples = []

indices = range(1,41)
for idx in indices:
    filename = modality + "_" + "dice" + str(idx) + ".xls"
    input_file = os.path.join(input_folder, filename)
    data = pd.read_csv(input_file, sep="\t", header = None)
    scores = data.iloc[0,:8]
    all_scores.append(scores)
    sample_name = data.iloc[0,9]
    all_samples.append(sample_name)

np_scores = np.vstack(all_scores)
df = pd.DataFrame(np_scores, columns=["LV", "Epi", "RV", "LA", "RA", "Ao", "PA", "WH"])
df["sample"] = all_samples

df = df[["sample", "LV", "Epi", "RV", "LA", "RA", "Ao", "PA", "WH"]]

output_fn = modality + "_compiled.csv"
output_file = os.path.join(output_folder, output_fn)

print("Writing dataframe to ", output_file)
df.to_csv(output_file, index=False)