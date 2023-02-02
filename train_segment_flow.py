import os
import torch.optim.lr_scheduler
from src.segment_flow import SegmentFlow
from src.integrator import IntegrateRK4
from src.loss import SoftDiceLoss, average_chamfer_distance_between_meshes
from torch.nn import CrossEntropyLoss
from src.dataset import ImageSegmentationDataset
from src.io_utils import SaveBestModel
from src.template import Template, BatchTemplate
import math
import yaml
import argparse
import numpy as np
from torch.utils.data import DataLoader

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def evaluate_model(net, integrator, dataset, batched_template, loss_config):
    assert len(dataset) > 0.0
    avg_loss = np.zeros(4)

    dice_evaluator = SoftDiceLoss()
    cross_entropy_evaluator = CrossEntropyLoss()

    print("\n\n\tEVALUATING TEST LOSS\n\n")
    batched_verts = batched_template.batch_vertex_coordinates()
    norm_type = loss_config["norm_type"]

    net.eval()
    for (idx, data) in enumerate(dataset):
        image = data["image"].unsqueeze(0).to(device).to(memory_format=torch.channels_last_3d)
        gt_segmentation = data["segmentation"].to(device)
        gt_meshes = [m.to(device) for m in data["meshes"]]

        with torch.no_grad():
            pred_segmentation, flow = net(image)

        deformed_verts = [integrator.integrate(flow, x) for x in batched_verts]
        batched_template.update_batched_vertices(deformed_verts, detach=False)

        chd, chn = average_chamfer_distance_between_meshes(batched_template.meshes_list, gt_meshes, norm_type)
        dice = dice_evaluator(pred_segmentation, gt_segmentation)
        cross_entropy = cross_entropy_evaluator(pred_segmentation, gt_segmentation)

        avg_loss[0] += dice.item() * loss_config["dice"]
        avg_loss[1] += cross_entropy.item() * loss_config["cross_entropy"]
        avg_loss[2] += chd.item() * loss_config["chamfer_distance"]
        avg_loss[3] += chn.item() * loss_config["chamfer_normal"]

    num_data_points = len(dataset)
    avg_loss /= num_data_points
    total = avg_loss.sum()

    out_str = "\tDice {:1.3e} | CE {:1.3e} | CHD {:1.3e} | CHN {:1.3e} | TOT {:1.3e}".format(
        avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3], total
    )
    print(out_str)
    print("\n\n")

    net.train()

    return total


def step_training_epoch(
        epoch,
        net,
        integrator,
        optimizer,
        scheduler,
        dataloader,
        validation_dataset,
        template,
        loss_config,
        save_best_model,
        eval_frequency=0.1
):
    assert len(dataloader) > 0
    assert len(validation_dataset) > 0

    avg_train_loss = avg_validation_loss = 0.0
    eval_counter = 0
    eval_every = int(math.ceil(eval_frequency * len(dataloader)))

    norm_type = loss_config["norm_type"]
    print("USING NORM TYPE : ", norm_type)

    dice_evaluator = SoftDiceLoss()
    cross_entropy_evaluator = CrossEntropyLoss()

    dice_weight = loss_config["dice"]
    cross_entropy_weight = loss_config["cross_entropy"]
    chamfer_distance_weight = loss_config["chamfer_distance"]
    chamfer_normal_weight = loss_config["chamfer_normal"]

    for (idx, data) in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)

        image = data["image"].to(device).to(memory_format=torch.channels_last_3d)
        gt_meshes = [m.to(device) for m in data["meshes"]]
        gt_segmentation = data["segmentation"].squeeze(1).to(device)

        batch_size = image.shape[0]
        batched_template = BatchTemplate.from_single_template(template, batch_size)
        batched_verts = batched_template.batch_vertex_coordinates()

        pred_segmentation, flow = net(image)

        deformed_verts = [integrator.integrate(flow, x) for x in batched_verts]
        batched_template.update_batched_vertices(deformed_verts, detach=False)

        chd, chn = average_chamfer_distance_between_meshes(batched_template.meshes_list, gt_meshes, norm_type)
        chd *= chamfer_distance_weight
        chn *= chamfer_normal_weight

        dice = dice_evaluator(pred_segmentation, gt_segmentation) * dice_weight
        cross_entropy = cross_entropy_evaluator(pred_segmentation, gt_segmentation) * cross_entropy_weight

        loss = chd + chn + dice + cross_entropy
        loss.backward()
        optimizer.step()

        avg_train_loss += loss.item()
        lr = optimizer.param_groups[0]["lr"]

        out_str = "\tBatch {:04d} | Dice {:1.3e} | CE {:1.3e} | CHD {:1.3e} | CHN {:1.3e} | TOT {:1.3e} | LR {:1.3e}".format(
            idx, dice.item(), cross_entropy.item(), chd.item(), chn.item(), loss.item(), lr
        )
        print(out_str)

        if (idx + 1) % eval_every == 0:
            eval_counter += 1
            batched_template = BatchTemplate.from_single_template(template, 1)
            validation_loss = evaluate_model(net, integrator, validation_dataset, batched_template, loss_config)
            avg_validation_loss += validation_loss
            save_best_model(validation_loss, epoch, net, integrator, optimizer)
            scheduler.step(validation_loss)

    avg_train_loss /= len(dataloader)
    if eval_counter > 0:
        avg_validation_loss /= eval_counter

    return avg_train_loss, avg_validation_loss


def run_training_loop(
        net,
        integrator,
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
            integrator,
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

    train_dataset = ImageSegmentationDataset(train_folder)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validation_folder = config["data"]["validation_folder"]
    validation_dataset = ImageSegmentationDataset(validation_folder)

    tmplt_fn = config["data"]["template_filename"]
    template = Template.from_vtk(tmplt_fn, device=device)

    net_config = config["model"]
    net = SegmentFlow.from_dict(net_config)
    net.to(device)
    integrator = IntegrateRK4()

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
    loss_config = config["loss"]

    train_loss, test_loss = run_training_loop(
        net,
        integrator,
        optimizer,
        scheduler,
        train_dataloader,
        validation_dataset,
        template,
        loss_config,
        save_best_model,
        num_epochs
    )

    output_data = np.array([train_loss, test_loss]).T
    df_outfile = os.path.join(output_dir, "loss_history.csv")
    np.savetxt(df_outfile, output_data, delimiter=",", header="train loss, test loss")
