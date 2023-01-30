import os
import torch.optim.lr_scheduler
from src.unet_segment import UnetSegment
from src.loss import DiceLoss
from torch.nn import CrossEntropyLoss
from src.dataset import ImageSegmentationDataset
from src.io_utils import SaveBestModel
import math
import yaml
import argparse
import numpy as np
from torch.utils.data import DataLoader


device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def evaluate_model(net, dataset, loss_weights):
    assert len(dataset) > 0
    avg_dice_loss = 0.0
    avg_cross_entropy_loss = 0.0

    dice_loss_evaluator = DiceLoss()
    cross_entropy_loss_evaluator = CrossEntropyLoss()

    print("\n\nEVALUATING TEST LOSS\n\n")

    net.eval()
    for (idx, data) in enumerate(dataset):
        image = data["image"].unsqueeze(0).to(device).to(memory_format=torch.channels_last_3d)
        segmentation = data["segmentation"].to(device)

        with torch.no_grad():
            prediction = net(image)
        
        dice = dice_loss_evaluator(prediction, segmentation)
        cross_entropy = cross_entropy_loss_evaluator(prediction, segmentation)

        avg_dice_loss += dice.item()
        avg_cross_entropy_loss += cross_entropy.item()
    
    avg_dice_loss /= len(dataset)
    avg_cross_entropy_loss /= len(dataset)
    total = loss_weights["dice"]*avg_dice_loss + loss_weights["cross_entropy"]*avg_cross_entropy_loss

    out_str = "\tDice {:1.3e} | Cross-Entropy {:1.3e} | Total {:1.3e}".format(
        avg_dice_loss,
        avg_cross_entropy_loss,
        total
    )
    print(out_str)
    print("\n\n")

    net.train()

    return total


def step_training_epoch(epoch, net, optimizer, scheduler, dataloader, validation_dataset, 
                        loss_weights, save_best_model, eval_frequency=0.01):
    
    assert len(dataloader) > 0
    assert len(validation_dataset) > 0
    assert eval_frequency > 0 and eval_frequency < 1

    avg_train_loss = 0.0
    avg_validation_loss = 0.0

    eval_every = int(math.ceil(eval_frequency * len(dataloader)))

    dice_loss_evaluator = DiceLoss()
    cross_entropy_loss_evaluator = CrossEntropyLoss(reduction="mean")
    dice_weight = loss_weights["dice"]
    cross_entropy_weight = loss_weights["cross_entropy"]
    eval_counter = 0

    for (idx, data) in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)

        image = data["image"].to(device).to(memory_format=torch.channels_last_3d)
        gt_segmentation = data["segmentation"].squeeze(1).to(device)

        prediction = net(image)

        dice = dice_loss_evaluator(prediction, gt_segmentation)
        cross_entropy = cross_entropy_loss_evaluator(prediction, gt_segmentation)

        loss = dice_weight*dice + cross_entropy_weight*cross_entropy
        loss.backward()
        optimizer.step()

        avg_train_loss += dice.item()

        lr = optimizer.param_groups[0]["lr"]
        out_str = "\tBatch {:04d} | Dice {:1.3e} | Cross-Entropy {:1.3e} | Total {:1.3e} | LR {:1.3e}".format(
            idx,
            dice.item(),
            cross_entropy.item(),
            loss.item(),
            lr
        )
        print(out_str)

        if (idx+1) % eval_every == 0:
            eval_counter += 1
            validation_loss = evaluate_model(net, validation_dataset, loss_weights)
            save_best_model(validation_loss, epoch, net, optimizer, "total_loss")
            avg_validation_loss += validation_loss
            scheduler.step(validation_loss)

    
    avg_train_loss /= len(dataloader)
    if eval_counter > 0:
        avg_validation_loss /= eval_counter

    return avg_train_loss, avg_validation_loss



def run_training_loop(net, optimizer, scheduler, dataloader, validation_dataset, loss_weights, num_epochs, save_best_model):
    train_loss = []
    validation_loss = []

    for epoch in range(num_epochs):
        print("\n\nSTARTING EPOCH {:03d}\n\n".format(epoch))
        avg_train_loss, avg_validation_loss = step_training_epoch(epoch, net, optimizer, scheduler, dataloader,
        validation_dataset, loss_weights, save_best_model)

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
    parser = argparse.ArgumentParser(description="Train Unet segmentor")
    parser.add_argument("-config", help="path to config file")
    args = parser.parse_args()

    config_fn = args.config
    config = get_config(config_fn)

    train_folder = config["data"]["train_folder"]
    batch_size = config["train"]["batch_size"]

    train_dataset = ImageSegmentationDataset(train_folder)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
    validation_folder = config["data"]["validation_folder"]
    validation_dataset = ImageSegmentationDataset(validation_folder)

    net_config = config["model"]
    net = UnetSegment.from_dict(net_config)
    net.to(device)

    optimizer_config = config["train"]["optimizer"]
    lr = optimizer_config["lr"]
    weight_decay = optimizer_config["weight_decay"]
    optimizer = torch.optim.Adam(net.parameters(), **optimizer_config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", **config["train"]["scheduler"])

    default_output_dir = os.path.dirname(config_fn)
    output_dir = config["data"].get("output_dir", default_output_dir)
    print("WRITING OUTPUT TO : ", output_dir, "\n\n")
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    best_model_fn = os.path.join(output_dir, "best_model_dict.pth")
    save_best_model = SaveBestModel(best_model_fn)

    num_epochs = config["train"]["num_epochs"]
    loss_weights = config["loss"]

    
    train_loss, test_loss = run_training_loop(net, optimizer, scheduler, train_dataloader, validation_dataset, loss_weights,
                                                num_epochs, save_best_model)