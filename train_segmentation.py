import torch.optim.lr_scheduler
from src.unet_segment import *
from src.flow_loss import *
from src.dataset import *
from src.io_utils import SaveBestModel, loss2str
import math
import yaml
import argparse
from collections import defaultdict

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def evaluate_model(net, dataset, loss_evaluators, loss_config):
    assert len(dataset) > 0
    running_validation_loss = defaultdict(float)

    for (idx, data) in enumerate(dataset):
        image = data["image"].unsqueeze(0).to(device).to(memory_format=torch.channels_last_3d)
        segmentation = data["segmentation"].unsqueeze(0).to(device)
        ground_truth = {"segmentation": segmentation}

        with torch.no_grad():
            prediction = net.predict(image)

        loss_components = compute_loss_components(prediction, ground_truth, loss_evaluators, loss_config)

        for (k, v) in loss_components.items():
            running_validation_loss[k] += v.item()

    num_data_points = len(dataset)
    for (k, v) in running_validation_loss.items():
        running_validation_loss[k] /= num_data_points

    return running_validation_loss


def step_training_epoch(
        epoch,
        net,
        optimizer,
        scheduler,
        dataloader,
        validation_dataset,
        loss_config,
        checkpointer,
        eval_frequency
):
    assert len(dataloader) > 0
    assert len(validation_dataset) > 0
    assert 0 < eval_frequency < 0.1

    running_training_loss = defaultdict(float)
    running_validation_loss = defaultdict(float)

    eval_every = int(math.ceil(eval_frequency * len(dataloader)))

    loss_evaluators = get_loss_evaluators(loss_config)
    eval_counter = 0

    for (idx, data) in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)

        image = data["image"].to(device).to(memory_format=torch.channels_last_3d)
        gt_segmentation = data["segmentation"].to(device)
        ground_truth = {"segmentation": gt_segmentation}

        prediction = net.predict(image)

        loss_components = compute_loss_components(prediction, ground_truth, loss_evaluators, loss_config)

        loss = loss_components["total"]
        loss.backward()
        optimizer.step()

        for (k, v) in loss_components.items():
            running_training_loss[k] += v.item()

        lr = optimizer.param_groups[0]["lr"]
        out_str = loss2str(loss_components)
        out_str = "\tBatch {:04d} | ".format(idx) + out_str + "LR {:1.3e}".format(lr)
        print(out_str)

        if (idx + 1) % eval_every == 0:
            print("\n\n\tEVALUATING MODEL")
            eval_counter += 1
            validation_loss_components = evaluate_model(net, validation_dataset, loss_evaluators, loss_config)
            out_str = "\t\t" + loss2str(validation_loss_components) + "\n\n"
            print(out_str)
            total_validation_loss = validation_loss_components["total"]
            save_data = {"model": net, "optimizer": optimizer}
            checkpointer.save_best_model(total_validation_loss, epoch, save_data)
            scheduler.step(total_validation_loss)

            for (k, v) in validation_loss_components.items():
                running_validation_loss[k] += v

    num_data_points = len(dataloader)

    for (k, v) in running_training_loss.items():
        running_training_loss[k] /= num_data_points

    if eval_counter > 0:
        for (k, v) in running_validation_loss.items():
            running_validation_loss[k] /= eval_counter

    return running_training_loss, running_validation_loss


def run_training_loop(
        net,
        optimizer,
        scheduler,
        dataloader,
        validation_dataset,
        loss_config,
        num_epochs,
        save_best_model
):
    train_loss = defaultdict(list)
    validation_loss = defaultdict(list)

    for epoch in range(num_epochs):
        print("\n\nSTARTING EPOCH {:03d}\n\n".format(epoch))
        avg_train_loss, avg_validation_loss = step_training_epoch(
            epoch,
            net,
            optimizer,
            scheduler,
            dataloader,
            validation_dataset,
            loss_config,
            save_best_model,
            eval_frequency
        )

        for (k, v) in avg_train_loss.items():
            train_loss[k].append(v)

        for (k, v) in avg_validation_loss.items():
            validation_loss[k].append(v)

        print("\n\n\t\tEPOCH {:03d}".format(epoch))
        out_str = "AVG TRAIN LOSS : \t" + loss2str(avg_train_loss)
        print(out_str)
        out_str = "AVG VALID LOSS : \t" + loss2str(avg_validation_loss)
        print(out_str)

        print("\n\nWRITING LOSS DATA\n\n")
        save_best_model.save_loss(train_loss, "train_loss.csv")
        save_best_model.save_loss(validation_loss, "validation_loss.csv")

    return train_loss, validation_loss


def get_config(config_fn):
    with open(config_fn, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unet segmentor")
    parser.add_argument("-config", help="path to config file")
    args = parser.parse_args()

    config_fn = args.config
    config = get_config(config_fn)

    train_folder = config["data"]["train_folder"]
    batch_size = config["train"]["batch_size"]

    train_dataloader = image_segmentation_mesh_dataloader(train_folder, shuffle=True, batch_size=batch_size)

    validation_folder = config["data"]["validation_folder"]
    validation_dataset = ImageSegmentationMeshDataset(validation_folder)

    net_config = config["model"]
    net = UnetSegment.from_dict(net_config)
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

    train_loss, validation_loss = run_training_loop(
        net,
        optimizer,
        scheduler,
        train_dataloader,
        validation_dataset,
        loss_config,
        save_best_model,
        num_epochs,
        eval_frequency
    )

    print("\n\nCOMPLETED TRAINING MODEL")
