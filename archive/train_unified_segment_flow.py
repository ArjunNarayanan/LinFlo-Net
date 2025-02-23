import os
import torch.optim.lr_scheduler
from src.segment_flow import *
from src.integrator import IntegrateFlowDivRK4
from src.flow_loss import *
from src.dataset import ImageSegmentationMeshDataset, image_segmentation_mesh_dataloader
from src.io_utils import SaveBestModel, load_yaml_config
from src.template import Template
import argparse
import train_workflow as workflow

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def initialize_model(model_config):
    pretrained_encoder = torch.load(model_config["pretrained_encoder"], map_location=device)
    pretrained_linear_transform = torch.load(model_config["pretrained_linear_transform"], map_location=device)
    encoder = Unet.from_dict(model_config["encoder"])

    decoder_input_channels = model_config["encoder"]["uparm_channels"][-1]
    decoder_hidden_channels = model_config["decoder"]["hidden_channels"]
    decoder_output_channels = model_config["decoder"]["output_channels"]
    unified_decoder = UnifiedDecoder(decoder_input_channels, decoder_hidden_channels, decoder_output_channels)

    flow_clip_value = model_config["clip_flow"]
    integrator = IntegrateFlowDivRK4(model_config["integrator"]["num_steps"])
    net = UnifiedSegmentFlow(INPUT_SHAPE,
                            pretrained_encoder,
                            pretrained_linear_transform,
                            encoder,
                            unified_decoder,
                            integrator,
                            flow_clip_value)
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Direct Segmentation Flow")
    parser.add_argument("-config", help="path to config file")
    args = parser.parse_args()

    config_fn = args.config
    config = load_yaml_config(config_fn)

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
    output_dir = config["data"].get("output_folder", default_output_dir)
    print("WRITING OUTPUT TO : ", output_dir, "\n\n")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    save_best_model = SaveBestModel(output_dir)

    num_epochs = config["train"]["num_epochs"]
    loss_config = config["loss"]

    train_loss, validation_loss = workflow.run_training_loop(net,
                                                             optimizer,
                                                             scheduler,
                                                             train_dataloader,
                                                             validation_dataset,
                                                             template,
                                                             loss_config,
                                                             save_best_model,
                                                             num_epochs)

    print("\n\nCOMPLETED TRAINING MODEL")
