from src.linear_transform import LinearTransformNet, LinearTransformWithEncoder
from src.dataset import ImageSegmentationMeshDataset, image_segmentation_mesh_dataloader
from src.template import Template, BatchTemplate
from src.loss import average_chamfer_distance_between_meshes
import math
import yaml
import argparse
import numpy as np
from src.io_utils import SaveBestModel
import torch
import os

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def evaluate_test_loss(net, dataset, batched_template, norm_type):
    assert len(dataset) > 0

    avg_loss = 0
    tmplt_coords = batched_template.batch_vertex_coordinates()

    print("\n\n\tEVALUATING TEST LOSS\n\n")

    net.eval()
    for (idx, data) in enumerate(dataset):
        img = data["image"].unsqueeze(0).to(memory_format=torch.channels_last_3d).to(device)

        with torch.no_grad():
            deformed_coords = net(img, tmplt_coords)

        batched_template.update_batched_vertices(deformed_coords, detach=False)
        gt_meshes = [m.to(device) for m in data["meshes"]]

        chmf_loss, _ = average_chamfer_distance_between_meshes(batched_template.meshes_list, gt_meshes, norm_type)

        avg_loss += chmf_loss

    net.train()

    avg_loss = avg_loss / len(dataset)

    out_str = "\tCHD {:1.3e}\n\n".format(avg_loss)
    print(out_str)

    return avg_loss


def step_training_epoch(epoch,
                        net,
                        optimizer,
                        scheduler,
                        dataloader,
                        validation_dataset,
                        template,
                        loss_config,
                        save_best_model,
                        eval_frequency=0.1):
    assert len(dataloader) > 0
    assert len(validation_dataset) > 0

    avg_train_loss = 0
    avg_test_loss = 0

    test_counter = 0

    eval_every = int(math.ceil(eval_frequency * len(dataloader)))
    norm_type = loss_config["norm_type"]
    print("USING NORM TYPE : ", norm_type)

    for (idx, data) in enumerate(dataloader):
        optimizer.zero_grad()

        image = data["image"].to(memory_format=torch.channels_last_3d).to(device)
        batch_size = image.shape[0]

        batched_template = BatchTemplate.from_single_template(template, batch_size)
        batched_verts = batched_template.batch_vertex_coordinates()

        deformed_verts = net(image, batched_verts)

        batched_template.update_batched_vertices(deformed_verts, detach=False)
        gt_meshes = [m.to(device) for m in data["meshes"]]

        chamfer_loss, _ = average_chamfer_distance_between_meshes(batched_template.meshes_list, gt_meshes, norm_type)
        total_loss = chamfer_loss

        total_loss.backward()
        optimizer.step()

        loss_val = total_loss.item()
        avg_train_loss += loss_val
        lr = optimizer.param_groups[0]["lr"]

        print(
            "\t\tMinibatch {:04d} | CHD {:1.3e} | LR {:1.3e}".format(idx, loss_val, lr))

        if (idx + 1) % eval_every == 0:
            batched_template = BatchTemplate.from_single_template(template, 1)
            test_chamfer_loss = evaluate_test_loss(net, validation_dataset, batched_template, norm_type)
            test_loss = test_chamfer_loss.item()
            avg_test_loss += test_loss
            test_counter += 1

            save_data = {"model": net, "optimizer": optimizer}
            save_best_model(test_loss, epoch, save_data)
            scheduler.step(test_loss)

    avg_train_loss = avg_train_loss / len(dataloader)
    if test_counter > 0:
        avg_test_loss = avg_test_loss / test_counter

    return avg_train_loss, avg_test_loss


def run_training_loop(
        net,
        optimizer,
        scheduler,
        dataloader,
        validation_dataset,
        template,
        loss_config,
        save_best_model,
        num_epochs,
):
    train_loss = []
    validation_loss = []

    for epoch in range(num_epochs):
        print("\n\nSTARTING EPOCH {:03d}\n\n".format(epoch))
        avg_train_loss, avg_validation_loss = step_training_epoch(
            epoch,
            net,
            optimizer,
            scheduler,
            dataloader,
            validation_dataset,
            template,
            loss_config,
            save_best_model
        )
        train_loss.append(avg_train_loss)
        validation_loss.append(avg_validation_loss)

        out_str = "\n\nEpoch {:03d} | Train Loss {:1.3e} | Validation Loss {:1.3e}\n\n".format(
            epoch,
            avg_train_loss,
            avg_validation_loss
        )
        print(out_str)

    return train_loss, validation_loss


def get_config(config_fn):
    with open(config_fn, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Segmentation Flow")
    parser.add_argument("-config", help="path to config file")
    args = parser.parse_args()

    config_fn = args.config
    config = get_config(config_fn)

    train_folder = config["data"]["train_folder"]
    batch_size = config["train"]["batch_size"]

    train_dataloader = image_segmentation_mesh_dataloader(train_folder, batch_size=batch_size, shuffle=True)

    validation_folder = config["data"]["validation_folder"]
    validation_dataset = ImageSegmentationMeshDataset(validation_folder)

    tmplt_fn = config["data"]["template_filename"]
    template = Template.from_vtk(tmplt_fn, device=device)

    net_config = config["model"]
    net = LinearTransformNet.from_dict(net_config)
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

    best_model_fn = os.path.join(output_dir, "best_model_dict.pth")
    save_best_model = SaveBestModel(best_model_fn)

    num_epochs = config["train"]["num_epochs"]
    loss_config = config["loss"]

    train_loss, test_loss = run_training_loop(net, optimizer, scheduler, train_dataloader,
                                              validation_dataset, template, loss_config, save_best_model, num_epochs)

    output_data = np.array([train_loss, test_loss]).T
    df_outfile = os.path.join(output_dir, "loss_history.csv")
    np.savetxt(df_outfile, output_data, delimiter=",", header="train loss, test loss")
