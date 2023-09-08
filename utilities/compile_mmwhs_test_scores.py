import pandas as pd
import os
import numpy as np
import argparse


def compile_scores():
    all_scores = []
    all_samples = []

    indices = range(1, 41)
    for idx in indices:
        filename = modality + "_" + "dice" + str(idx) + ".xls"
        input_file = os.path.join(input_folder, filename)
        data = pd.read_csv(input_file, sep="\t", header=None)
        scores = data.iloc[0, :8]
        all_scores.append(scores)
        sample_name = data.iloc[0, 9]
        all_samples.append(sample_name)

    np_scores = np.vstack(all_scores)
    df = pd.DataFrame(np_scores, columns=["LV", "Epi", "RV", "LA", "RA", "Ao", "PA", "WH"])
    df["sample"] = all_samples

    df = df[["sample", "LV", "Epi", "RV", "LA", "RA", "Ao", "PA", "WH"]]

    output_fn = modality + "_compiled.csv"
    output_file = os.path.join(output_folder, output_fn)

    print("Writing dataframe to ", output_file)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile MMWHS test dice scores into one file")
    parser.add_argument("-input", help="Input folder", required=True)
    parser.add_argument("-output", help="Output folder", default=None)
    parser.add_argument("-modality", help="Modality", required=True)
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output
    if output_folder is None:
        output_folder = os.path.dirname(input_folder)

    modality = args.modality
    compile_scores()
