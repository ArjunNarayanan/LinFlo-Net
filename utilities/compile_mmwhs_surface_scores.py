import os
import pandas as pd
import argparse


def structure_to_output_name(name):
    if name == "LO":
        return "Ao"
    else:
        return name


def compile_results_for_structure(structure):
    vals = []
    for idx in range(1, 41):
        filename = modality + "_surface" + str(idx) + "_" + structure + ".xls"
        # print("Processing : ", filename)
        filepath = os.path.join(input_folder, filename)
        df = pd.read_csv(filepath, sep="\t", usecols=[1, 2, 3], skiprows=2, header=None)
        vals.append(df.iloc[0].to_list())

    output_df = pd.DataFrame(vals, columns=["Avg", "Std", "Max"])

    return output_df


def update_summary_statistics(df):
    avg = df.iloc[:, 1:].mean(axis=0)
    median = df.iloc[:, 1:].median(axis=0)
    maximum = df.iloc[:, 1:].max(axis=0)

    avg["sample"] = "Average"
    median["sample"] = "Median"
    maximum["sample"] = "Max"

    numrows = len(df)
    df.loc[numrows] = avg
    df.loc[numrows+1] = median
    df.loc[numrows+2] = maximum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile surface distance metrics")
    parser.add_argument("-input", help="Input folder", required=True)
    parser.add_argument("-modality", help="Imaging modality", required=True)
    parser.add_argument("-output", help="Output folder", default=None)
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output
    if output_folder is None:
        output_folder = os.path.dirname(input_folder)

    modality = args.modality

    csv_structure_names = ["LV", "Epi", "RV", "LA", "RA", "LO", "PA", "WHS"]
    output_names = ["LV", "Epi", "RV", "LA", "RA", "Ao", "PA", "WH"]

    sample_names = [modality + str(2000 + idx) for idx in range(1, 41)]
    assd_df = pd.DataFrame(sample_names, columns=["sample"])
    std_df = pd.DataFrame(sample_names, columns=["sample"])
    max_df = pd.DataFrame(sample_names, columns=["sample"])

    for idx, name in enumerate(csv_structure_names):
        output_structure_name = output_names[idx]
        print("Compiling results for structure ", output_structure_name)
        structure_df = compile_results_for_structure(name)
        assd_df[output_structure_name] = structure_df["Avg"]
        std_df[output_structure_name] = structure_df["Std"]
        max_df[output_structure_name] = structure_df["Max"]

    output_order = ["sample", "Epi", "LA", "LV", "RA", "RV", "Ao", "PA", "WH"]
    assd_df = assd_df[output_order]
    std_df = std_df[output_order]
    max_df = max_df[output_order]

    update_summary_statistics(assd_df)
    update_summary_statistics(std_df)
    update_summary_statistics(max_df)

    print("\n\nWriting compiled results:")
    assd_output_file = os.path.join(output_folder, modality + "_assd.csv")
    assd_df.to_csv(assd_output_file, index=False)
    std_output_file = os.path.join(output_folder, modality + "_surface_std.csv")
    std_df.to_csv(std_output_file, index=False)
    max_output_file = os.path.join(output_folder, modality + "_hausdorff.csv")
    max_df.to_csv(max_output_file, index=False)
