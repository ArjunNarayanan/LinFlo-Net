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
    output_file_name = modality + "_surface_" + structure_to_output_name(structure) + ".csv"

    output_filepath = os.path.join(os.path.dirname(input_folder), output_file_name)

    return output_df


input_folder = "output/WholeHeartData/ct-mr-cropped/LT-flow/udf-1-contd/MMWHS-test/test-results"
output_folder = os.path.dirname(input_folder)

modality = "ct"
csv_structure_names = ["LV", "Epi", "RV", "LA", "RA", "LO", "PA", "WHS"]

assd_df = pd.DataFrame()
std_df = pd.DataFrame()
max_df = pd.DataFrame()

LV_df = compile_results_for_structure("LV")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Compile surface distance metrics")
#     parser.add_argument("-input", help="Input folder", required=True)
#     parser.add_argument("-modality", help="Imaging modality", required=True)
#     parser.add_argument("-output", help="Output folder", default=None)
#     args = parser.parse_args()
#
#     input_folder = args.input
#     output_folder = args.output
#     if output_folder is None:
#         output_folder = os.path.dirname(input_folder)
#
#     modality = args.modality
#
#     csv_structure_names = ["LV", "Epi", "RV", "LA", "RA", "LO", "PA", "WHS"]
#     for structure in csv_structure_names:
#         compile_results_for_structure(structure)
