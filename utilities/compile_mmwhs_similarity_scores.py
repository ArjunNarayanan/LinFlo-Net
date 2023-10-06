import pandas as pd
import os
import numpy as np
import argparse


def update_summary_statistics(df):
    avg = df.iloc[:, 1:].mean(axis=0)
    median = df.iloc[:, 1:].median(axis=0)
    maximum = df.iloc[:, 1:].min(axis=0)

    avg["sample"] = "Average"
    median["sample"] = "Median"
    maximum["sample"] = "Min"

    numrows = len(df)
    df.loc[numrows] = avg
    df.loc[numrows + 1] = median
    df.loc[numrows + 2] = maximum


def compile_dice_scores():
    print("\nCompiling dice scores : ")
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

    df = df[output_order]

    update_summary_statistics(df)

    output_fn = modality + "_dice_compiled.csv"
    output_file = os.path.join(output_folder, output_fn)

    print("Writing dataframe to ", output_file)
    df.to_csv(output_file, index=False)


def compile_jaccard_scores():
    print("\nCompiling Jaccard Index :")
    all_scores = []
    all_samples = []

    indices = range(1, 41)
    for idx in indices:
        filename = modality + "_" + "jaccard" + str(idx) + ".xls"
        input_file = os.path.join(input_folder, filename)
        data = pd.read_csv(input_file, sep="\t", header=None)
        scores = data.iloc[0, :8]
        all_scores.append(scores)
        sample_name = data.iloc[0, 9]
        all_samples.append(sample_name)

    np_scores = np.vstack(all_scores)
    df = pd.DataFrame(np_scores, columns=["LV", "Epi", "RV", "LA", "RA", "Ao", "PA", "WH"])
    df["sample"] = all_samples

    df = df[output_order]
    update_summary_statistics(df)

    output_fn = modality + "_jaccard_compiled.csv"
    output_file = os.path.join(output_folder, output_fn)

    print("Writing dataframe to ", output_file)
    df.to_csv(output_file, index=False)


def compute_metrics(metric):
    if metric == "dice":
        compile_dice_scores()
    elif metric == "jaccard":
        compile_jaccard_scores()
    else:
        raise ValueError("Unexpected metric type : ", metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile MMWHS test dice scores into one file")
    parser.add_argument("-input", help="Input folder", required=True)
    parser.add_argument("-output", help="Output folder", default=None)
    parser.add_argument("-metrics", help="Metrics to compute", nargs="*", default=None)
    parser.add_argument("-modality", help="Modality", required=True)
    args = parser.parse_args()

    output_order = ["sample", "Epi", "LA", "LV", "RA", "RV", "Ao", "PA", "WH"]
    input_folder = args.input
    output_folder = args.output
    if output_folder is None:
        output_folder = os.path.dirname(input_folder)

    modality = args.modality
    metrics = args.metrics
    if metrics is None:
        metrics = ["dice", "jaccard"]

    for metric in metrics:
        compute_metrics(metric)
