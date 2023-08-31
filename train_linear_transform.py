from src.linear_transform import LinearTransformNet
from src.dataset import ImageSegmentationMeshDataset, image_segmentation_mesh_dataloader
from src.template import Template
import argparse
from src.io_utils import SaveBestModel, load_yaml_config
import torch
import os
import train_workflow as workflow

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def initialize_model(model_config):
    net = LinearTransformNet.from_dict(model_config)
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Direct Linear Transform")
    parser.add_argument("-config", help="path to config file")
    args = parser.parse_args()

    config_fn = args.config
    config = load_yaml_config(config_fn)

    train_folder = config["data"]["train_folder"]
    batch_size = config["train"]["batch_size"]

    train_dataloader = image_segmentation_mesh_dataloader(train_folder, batch_size=batch_size, shuffle=True)
    num_train_data = len(train_dataloader) * batch_size
    print("\nTRAIN DATASET SIZE : ", num_train_data)

    validation_folder = config["data"]["validation_folder"]
    validation_dataset = ImageSegmentationMeshDataset(validation_folder)
    num_validation_data = len(validation_dataset)
    print("\nVALIDATION DATASET SIZE : ", num_validation_data)

    tmplt_fn = config["data"]["template_filename"]
    template = Template.from_vtk(tmplt_fn, device=device)

    model_config = config["model"]
    net = initialize_model(model_config)
    net.to(device)

    optimizer_config = config["train"]["optimizer"]
    lr = optimizer_config["lr"]
    weight_decay = optimizer_config["weight_decay"]
    optimizer = torch.optim.Adam(net.parameters(), **optimizer_config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", **config["train"]["scheduler"])

    default_output_dir = os.path.dirname(config_fn)
    output_dir = config["data"].get("output_folder", default_output_dir)
    print("WRITING OUTPUT TO : ", output_dir, "\n\n")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    save_best_model = SaveBestModel(output_dir)

    num_epochs = config["train"]["num_epochs"]
    eval_frequency = config["train"].get("eval_frequency", 0.1)
    loss_config = config["loss"]

    train_loss, validation_loss = workflow.run_training_loop(net,
                                                             optimizer,
                                                             scheduler,
                                                             train_dataloader,
                                                             validation_dataset,
                                                             template,
                                                             loss_config,
                                                             save_best_model,
                                                             num_epochs,
                                                             eval_frequency,
                                                             point_cloud=None
                                                             )

    print("\n\nCOMPLETED TRAINING MODEL")
