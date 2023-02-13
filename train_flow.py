import os
import torch.optim.lr_scheduler
from src.flow import EncodeLinearTransformFlow, Flow, FlowDiv
from src.utilities import batch_occupancy_map_from_vertices
from src.integrator import IntegrateFlowDivRK4
from src.loss import average_chamfer_distance_between_meshes
from src.dataset import ImageSegmentationMeshDataset, image_segmentation_mesh_dataloader
from src.io_utils import SaveBestModel
from src.template import Template, BatchTemplate
import math
import yaml
import argparse
import numpy as np

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def mean_divergence_loss(div_integral):
    mean_divergence = [(-fd).exp().mean() for fd in div_integral]
    mean_divergence = sum(mean_divergence) / len(mean_divergence)
    return mean_divergence


def evaluate_model(net, dataset, batched_template, loss_config):
    assert len(dataset) > 0.0
    avg_loss = np.zeros(3)

    print("\n\n\tEVALUATING TEST LOSS\n\n")
    batched_verts = batched_template.batch_vertex_coordinates()
    norm_type = loss_config["norm_type"]
    flow_div = FlowDiv(net.flow.input_shape)

    net.eval()
    for (idx, data) in enumerate(dataset):
        image = data["image"].unsqueeze(0).to(device).to(memory_format=torch.channels_last_3d)
        gt_meshes = [m.to(device) for m in data["meshes"]]

        encoding = net.encoder(image)
        encoding = torch.cat([image, encoding], dim=1)
        lt_deformed_vertices = net.linear_transform(encoding, batched_verts)
        occupancy = batch_occupancy_map_from_vertices(lt_deformed_vertices, 1, net.flow.input_shape)
        encoding - torch.cat([encoding, occupancy], dim=1)
        with torch.no_grad():
            flow = net.flow.get_flow_field(encoding)

        flow_and_div = flow_div.get_flow_div(flow)

        deformed_verts, div_integral = net.integrator.integrate_flow_and_div(flow_and_div, batched_verts)
        batched_template.update_batched_vertices(deformed_verts, detach=False)

        mean_divergence = mean_divergence_loss(div_integral)
        chd, chn = average_chamfer_distance_between_meshes(batched_template.meshes_list, gt_meshes, norm_type)

        avg_loss[0] += chd.item() * loss_config["chamfer_distance"]
        avg_loss[1] += chn.item() * loss_config["chamfer_normal"]
        avg_loss[2] += mean_divergence.item() * loss_config["divergence"]

    num_data_points = len(dataset)
    avg_loss /= num_data_points
    total = avg_loss.sum()

    out_str = "\tCHD {:1.3e} | CHN {:1.3e} | MD {:1.3e} | TOT {:1.3e}".format(
        avg_loss[0], avg_loss[1], avg_loss[2], total
    )
    print(out_str)
    print("\n\n")

    net.train()

    return total


def step_training_epoch(
        epoch,
        net,
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

    flow_div = FlowDiv(net.flow.input_shape)

    chamfer_distance_weight = loss_config["chamfer_distance"]
    chamfer_normal_weight = loss_config["chamfer_normal"]
    divergence_weight = loss_config["divergence"]

    for (idx, data) in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)

        image = data["image"].to(device).to(memory_format=torch.channels_last_3d)
        gt_meshes = [m.to(device) for m in data["meshes"]]

        batch_size = image.shape[0]
        batched_template = BatchTemplate.from_single_template(template, batch_size)
        batched_verts = batched_template.batch_vertex_coordinates()

        encoding = net.encoder(image)
        encoding = torch.cat([image, encoding], dim=1)
        lt_deformed_vertices = net.linear_transform(encoding, batched_verts)
        occupancy = batch_occupancy_map_from_vertices(lt_deformed_vertices, batch_size, net.flow.input_shape)
        encoding - torch.cat([encoding, occupancy], dim=1)
        flow = net.flow.get_flow_field(encoding)
        flow_and_div = flow_div.get_flow_div(flow)

        deformed_verts, div_integral = net.integrator.integrate_flow_and_div(flow_and_div, lt_deformed_vertices)
        batched_template.update_batched_vertices(deformed_verts, detach=False)

        divergence_loss = divergence_weight * mean_divergence_loss(div_integral)

        chd, chn = average_chamfer_distance_between_meshes(batched_template.meshes_list, gt_meshes, norm_type)
        chd *= chamfer_distance_weight
        chn *= chamfer_normal_weight

        loss = chd + chn + divergence_loss
        loss.backward()
        optimizer.step()

        avg_train_loss += loss.item()
        lr = optimizer.param_groups[0]["lr"]

        out_str = "\tBatch {:04d} | CHD {:1.3e} | CHN {:1.3e} | MD {:1.3e} | TOT {:1.3e} | LR {:1.3e}".format(
            idx, chd.item(), chn.item(), divergence_loss.item(), loss.item(), lr
        )
        print(out_str)

        if (idx + 1) % eval_every == 0:
            eval_counter += 1
            batched_template = BatchTemplate.from_single_template(template, 1)
            validation_loss = evaluate_model(net, validation_dataset, batched_template, loss_config)
            avg_validation_loss += validation_loss

            save_data = {"model": net, "optimizer": optimizer}
            save_best_model(validation_loss, epoch, save_data)
            scheduler.step(validation_loss)

    avg_train_loss /= len(dataloader)
    if eval_counter > 0:
        avg_validation_loss /= eval_counter

    return avg_train_loss, avg_validation_loss


def run_training_loop(net, optimizer, scheduler, dataloader, validation_dataset, template, loss_config,
                      save_best_model, num_epochs):
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

    model_config = config["model"]
    encoder = torch.load(model_config["pretrained_encoder"], map_location=device)
    linear_transform = torch.load(model_config["pretrained_linear_transform"], map_location=device)
    flow = Flow.from_dict(model_config["flow"])
    integrator = IntegrateFlowDivRK4(model_config["integrator"]["num_steps"])
    net = EncodeLinearTransformFlow(encoder, linear_transform, flow, integrator)
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
    loss_config = config["loss"]

    train_loss, test_loss = run_training_loop(net, optimizer, scheduler, train_dataloader,
                                              validation_dataset, template, loss_config, save_best_model, num_epochs)

    output_data = np.array([train_loss, test_loss]).T
    df_outfile = os.path.join(output_dir, "loss_history.csv")
    np.savetxt(df_outfile, output_data, delimiter=",", header="train loss, test loss")
