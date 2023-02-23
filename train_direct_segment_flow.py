import os
import torch.optim.lr_scheduler
from torch.nn import CrossEntropyLoss
from src.segment_flow import *
from src.utilities import batch_occupancy_map_from_vertices
from src.integrator import IntegrateFlowDivRK4
from src.flow_loss import *
from src.dataset import ImageSegmentationMeshDataset, image_segmentation_mesh_dataloader
from src.io_utils import SaveBestModel
from src.template import Template, BatchTemplate
import math
import yaml
import argparse
import numpy as np
from collections import defaultdict

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def get_model_predictions(net, image, template):
    flow_div = FlowDiv(INPUT_SHAPE)
    batch_size = image.shape[0]

    batched_template = BatchTemplate.from_single_template(template, batch_size)
    batched_verts = batched_template.batch_vertex_coordinates()

    lt_deformed_vertices = net.pretrained_linear_transform(image, batched_verts)
    occupancy = batch_occupancy_map_from_vertices(lt_deformed_vertices, batch_size, INPUT_SHAPE)
    encoder_input = net.get_encoder_input(image, occupancy)

    encoding = net.encoder(encoder_input)
    predicted_segmentation = net.segment_decoder(encoding)
    flow_field = net.flow_decoder(encoding)

    flow_and_div = flow_div.get_flow_div(flow_field)
    deformed_verts, div_integral = net.integrator.integrate_flow_and_div(flow_and_div, lt_deformed_vertices)
    batched_template.update_batched_vertices(deformed_verts, detach=False)

    predictions = {"meshes": batched_template.meshes_list,
                   "segmentation": predicted_segmentation,
                   "divergence_integral": div_integral}

    return predictions


def evaluate_model(net, dataset, template, loss_evaluators, loss_config):
    assert len(dataset) > 0.0
    running_validation_loss = defaultdict(float)

    for (idx, data) in enumerate(dataset):
        image = data["image"].unsqueeze(0).to(device).to(memory_format=torch.channels_last_3d)
        gt_meshes = [m.to(device) for m in data["meshes"]]
        gt_segmentation = data["segmentation"].unsqueeze(0).to(device)
        ground_truth = {"meshes": gt_meshes, "segmentation": gt_segmentation}

        with torch.no_grad():
            predictions = get_model_predictions(net, image, template)

        loss_components = compute_loss_components(predictions, ground_truth, loss_evaluators, loss_config)

        for (k, v) in loss_components.items():
            running_validation_loss[k] += v.item()

    num_data_points = len(dataset)
    for (k, v) in running_validation_loss.items():
        running_validation_loss[k] /= num_data_points

    return running_validation_loss


def loss2str(loss_components):
    out_str = ""
    if "chamfer_distance" in loss_components:
        out_str += "CHD {:1.3e} | CHN {:1.3e} | ".format(loss_components["chamfer_distance"],
                                                         loss_components["chamfer_normal"])
    if "divergence" in loss_components:
        out_str += "DIV {:1.3e} | ".format(loss_components["divergence"])
    if "cross_entropy" in loss_components:
        out_str += "MCE {:1.3e} | ".format(loss_components["cross_entropy"])
    if "dice" in loss_components:
        out_str += "DIC {:1.3e}".format(loss_components["dice"])
    if "edge" in loss_components:
        out_str += "EDG {:1.3e} | ".format(loss_components["edge"])
    if "laplace" in loss_components:
        out_str += "LAP {:1.3e} | ".format(loss_components["laplace"])
    if "normal" in loss_components:
        out_str += "NOR {:1.3e} | ".format(loss_components["normal"])

    return out_str


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

    eval_counter = 0
    eval_every = int(math.ceil(eval_frequency * len(dataloader)))

    norm_type = loss_config["norm_type"]
    print("USING NORM TYPE : ", norm_type)

    cross_entropy_evaluator = CrossEntropyLoss(reduction="mean")
    soft_dice_evaluator = SoftDiceLoss()
    loss_evaluators = {"cross_entropy": cross_entropy_evaluator,
                       "dice": soft_dice_evaluator}
    running_training_loss = defaultdict(float)
    running_validation_loss = defaultdict(float)

    for (idx, data) in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)

        image = data["image"].to(device).to(memory_format=torch.channels_last_3d)
        gt_meshes = [m.to(device) for m in data["meshes"]]
        gt_segmentation = data["segmentation"].to(device)
        ground_truth = {"meshes": gt_meshes, "segmentation": gt_segmentation}

        predictions = get_model_predictions(net, image, template)

        loss_components = compute_loss_components(predictions, ground_truth, loss_evaluators, loss_config)
        loss = sum(loss_components.values())
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
            validation_loss_components = evaluate_model(net, validation_dataset, template, loss_evaluators, loss_config)
            out_str = "\t\t" + loss2str(validation_loss_components) + "\n\n"
            print(out_str)
            validation_chd = validation_loss_components["chamfer_distance"] / loss_config["chamfer_distance"]
            save_data = {"model": net, "optimizer": optimizer}
            save_best_model(validation_chd, epoch, save_data)
            scheduler.step(validation_chd)

            for (k, v) in validation_loss_components.items():
                running_validation_loss[k] += v

    num_data_points = len(dataloader)
    for (k, v) in running_training_loss.items():
        running_training_loss[k] /= num_data_points

    if eval_counter > 0:
        for (k, v) in running_validation_loss.items():
            running_validation_loss[k] /= eval_counter

    return running_training_loss, running_validation_loss


def run_training_loop(net,
                      optimizer,
                      scheduler,
                      dataloader,
                      validation_dataset,
                      template,
                      loss_config,
                      save_best_model,
                      num_epochs):
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
            template,
            loss_config,
            save_best_model
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

    return train_loss, validation_loss


def get_config(config_fn):
    with open(config_fn, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def initialize_model(model_config):
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
    net = LinearTransformSegmentFlow(pretrained_linear_transform,
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

    train_loss, test_loss = run_training_loop(net,
                                              optimizer,
                                              scheduler,
                                              train_dataloader,
                                              validation_dataset,
                                              template,
                                              loss_config,
                                              save_best_model,
                                              num_epochs)

    output_data = np.array([train_loss, test_loss]).T
    df_outfile = os.path.join(output_dir, "loss_history.csv")
    np.savetxt(df_outfile, output_data, delimiter=",", header="train loss, test loss")
