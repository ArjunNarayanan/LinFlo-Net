import pandas as pd
import os
import numpy as np

input_folder = "output/segment_flow/model-7/test/mr/dice"
output_folder = "output/segment_flow/model-7/test/"

modality = "mr"
all_scores = []
all_samples = []

for idx in range(1,41):
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