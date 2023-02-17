import os
import torch.optim.lr_scheduler
from torch.nn import CrossEntropyLoss
from src.segment_flow import *
from src.utilities import batch_occupancy_map_from_vertices
from src.integrator import IntegrateFlowDivRK4
from src.loss import *
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
    avg_loss = 0.0

    print("\n\n\tEVALUATING TEST LOSS\n\n")
    batched_verts = batched_template.batch_vertex_coordinates()
    norm_type = loss_config["norm_type"]

    for (idx, data) in enumerate(dataset):
        image = data["image"].unsqueeze(0).to(device).to(memory_format=torch.channels_last_3d)
        gt_meshes = [m.to(device) for m in data["meshes"]]

        with torch.no_grad():
            deformed_verts = net(image, batched_verts)

        batched_template.update_batched_vertices(deformed_verts, detach=False)

        chd, _ = average_chamfer_distance_between_meshes(batched_template.meshes_list, gt_meshes, norm_type)

        avg_loss += chd

    num_data_points = len(dataset)
    avg_loss /= num_data_points

    out_str = "\tAVG. VALIDATION CHAMFER DISTANCE : {:1.3e}".format(avg_loss.item())
    print(out_str)
    print("\n\n")

    return avg_loss


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

    flow_div = FlowDiv(INPUT_SHAPE)
    cross_entropy_evaluator = CrossEntropyLoss(reduction="mean")

    chamfer_distance_weight = loss_config["chamfer_distance"]
    chamfer_normal_weight = loss_config["chamfer_normal"]
    divergence_weight = loss_config["divergence"]
    edge_loss_weight = loss_config["edge"]
    laplace_loss_weight = loss_config["laplace"]
    normal_consistency_loss_weight = loss_config["normal"]
    cross_entropy_weight = loss_config["cross_entropy"]

    for (idx, data) in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)

        image = data["image"].to(device).to(memory_format=torch.channels_last_3d)
        gt_meshes = [m.to(device) for m in data["meshes"]]
        gt_segmentation = data["segmentation"].to(device)

        batch_size = image.shape[0]
        batched_template = BatchTemplate.from_single_template(template, batch_size)
        batched_verts = batched_template.batch_vertex_coordinates()

        pre_encoding = net.pretrained_encoder(image)
        pre_encoding = torch.cat([image, pre_encoding], dim=1)
        lt_deformed_vertices = net.pretrained_linear_transform(pre_encoding, batched_verts)

        encoding = net.encoder(pre_encoding)
        segmentation = net.segment_decoder(encoding)

        occupancy = batch_occupancy_map_from_vertices(lt_deformed_vertices, batch_size, INPUT_SHAPE)
        encoding = torch.cat([encoding, occupancy], dim=1)
        flow_field = net.flow_decoder(encoding)
        flow_and_div = flow_div.get_flow_div(flow_field)

        deformed_verts, div_integral = net.integrator.integrate_flow_and_div(flow_and_div, lt_deformed_vertices)
        batched_template.update_batched_vertices(deformed_verts, detach=False)

        cross_entropy_loss = cross_entropy_weight * cross_entropy_evaluator(segmentation, gt_segmentation)
        divergence_loss = divergence_weight * mean_divergence_loss(div_integral)

        chd, chn = average_chamfer_distance_between_meshes(batched_template.meshes_list, gt_meshes, norm_type)
        chd *= chamfer_distance_weight
        chn *= chamfer_normal_weight

        edge_loss = average_mesh_edge_loss(batched_template.meshes_list) * edge_loss_weight
        laplace_loss = average_laplacian_smoothing_loss(batched_template.meshes_list) * laplace_loss_weight
        normal_loss = average_normal_consistency_loss(batched_template.meshes_list) * normal_consistency_loss_weight

        loss = chd + chn + divergence_loss + edge_loss + laplace_loss + normal_loss + cross_entropy_loss
        loss.backward()
        optimizer.step()

        avg_train_loss += chd.item()
        lr = optimizer.param_groups[0]["lr"]

        out_str = ("\tBatch {:04d} | CHD {:1.3e} | CHN {:1.3e} | MD {:1.3e} | ED {:1.3e} | " + \
                   "LP {:1.3e} | NR {:1.3e} | CE {:1.3e} | TOT {:1.3e} | LR {:1.3e}").format(
            idx, chd.item(), chn.item(), divergence_loss.item(), edge_loss.item(), laplace_loss.item(),
            normal_loss.item(), cross_entropy_loss.item(), loss.item(), lr
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


def initialize_model(model_config):
    pretrained_encoder = torch.load(model_config["pretrained_encoder"], map_location=device)
    pretrained_linear_transform = torch.load(model_config["pretrained_linear_transform"], map_location=device)
    encoder = Unet.from_dict(model_config["encoder"])

    decoder_input_channels = model_config["encoder"]["uparm_channels"][-1]
    decoder_hidden_channels = model_config["segment"]["decoder_hidden_channels"]
    decoder_output_channels = model_config["segment"]["output_channels"]
    segment_decoder = Decoder(decoder_input_channels, decoder_hidden_channels, decoder_output_channels)

    # since we add occupancy as a new channel, input channels increases by one
    decoder_input_channels = decoder_input_channels + 1
    decoder_hidden_channels = model_config["flow"]["decoder_hidden_channels"]
    flow_clip_value = model_config["flow"]["clip"]
    flow_decoder = FlowDecoder(decoder_input_channels, decoder_hidden_channels, flow_clip_value)

    integrator = IntegrateFlowDivRK4(model_config["integrator"]["num_steps"])
    net = EncodeLinearTransformSegmentFlow(pretrained_encoder,
                                           pretrained_linear_transform,
                                           encoder,
                                           segment_decoder,
                                           flow_decoder,
                                           integrator)
    return net


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
    INPUT_SHAPE = model_config["encoder"]["input_shape"]
    net = initialize_model(model_config)
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
