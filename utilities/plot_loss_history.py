import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def plot_loss(train_loss, validation_loss, loss_name, filename=None):
    fig, ax = plt.subplots()
    ax.plot(train_loss, label="train")
    ax.plot(validation_loss, label="validation")
    ax.set_xlabel("Epochs")
    ax.set_ylabel(loss_name)
    ax.grid()
    ax.legend()
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename)
        plt.close(fig)


def plot_all_losses(train_df, validation_df, output_dir):
    for loss_name in train_df.keys():
        assert loss_name in validation_df

        train_loss = train_df[loss_name]
        validation_loss = validation_df[loss_name]
        outfile = os.path.join(output_dir, loss_name + ".png")
        plot_loss(train_loss, validation_loss, loss_name, outfile)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="plot loss history from training logs")
    parser.add_argument("-input", help="input folder", required=True)
    args = parser.parse_args()

    root_dir = args.input
    validation_loss_file = os.path.join(root_dir, "validation_loss.csv")
    train_loss_file = os.path.join(root_dir, "train_loss.csv")
    assert os.path.isdir(root_dir)
    assert os.path.isfile(validation_loss_file)
    assert os.path.isfile(train_loss_file)

    output_dir = os.path.join(root_dir, "loss_plots")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    train_df = pd.read_csv(train_loss_file)
    validation_df = pd.read_csv(validation_loss_file)

    plot_all_losses(train_df, validation_df, output_dir)
