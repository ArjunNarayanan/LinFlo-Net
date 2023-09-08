import os
import pandas as pd
import numpy as np
import argparse

def perturb_coordinate(coordinate):
    lower_key = coordinate + "0"
    upper_key = coordinate + "1"
    diff_key = "d" + coordinate
    max_key = coordinate + "m"
    perturbed_X0 = (df[lower_key] - np.random.choice(margin_range, numsamples)).clip(lower=0)
    perturbed_X1 = (df[upper_key] + np.random.choice(margin_range, numsamples)).clip(upper=df[max_key])
    df[lower_key] = perturbed_X0
    df[upper_key] = perturbed_X1
    df[diff_key] = perturbed_X1 - perturbed_X0



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Add random margins to crop bounding box")
    parser.add_argument("-input", help="input CSV file with bounding box data", required=True)
    parser.add_argument("-output", help="output CSV file name", default="perturbed_crop_bounds.csv")
    parser.add_argument("-lower", help="minimum margin to add", default=30)
    parser.add_argument("-upper", help="maximum margin to add", default=50)
    args = parser.parse_args()

    input_csv = args.input
    output_csv = args.output
    lower_margin = args.lower
    upper_margin = args.upper

    margin_range = range(lower_margin, upper_margin)
    df = pd.read_csv(input_csv)
    original_df = df.copy(deep=True)
    numsamples = len(df)

    print("Perturbing X bounds:")
    perturb_coordinate("X")
    print("Perturbing Y bounds:")
    perturb_coordinate("Y")
    print("Perturbing Z bounds:")
    perturb_coordinate("Z")

    assert (df["X0"] >= 0).all()
    assert (df["Y0"] >= 0).all()
    assert (df["Z0"] >= 0).all()

    assert (df["X1"] <= df["Xm"]).all()
    assert (df["Y1"] <= df["Ym"]).all()
    assert (df["Z1"] <= df["Zm"]).all()

    print("Computing volume ratio:")
    cropped_vol = df["dX"] * df["dY"] * df["dZ"]
    original_vol = df["Xm"] * df["Ym"] * df["Zm"]
    df["volume_ratio"] = cropped_vol / original_vol

    output_dir = os.path.dirname(input_csv)
    output_filename = os.path.join(output_dir, output_csv)
    df.to_csv(output_filename, index=False)