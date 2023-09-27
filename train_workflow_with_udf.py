import torch.optim.lr_scheduler
from src.flow_loss import *
from src.io_utils import loss2str
import math
from src.template import BatchTemplate
from collections import defaultdict

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def evaluate_model(net, dataset, template, template_distance_map, loss_evaluators, loss_config):
    assert len(dataset) > 0.0
    running_validation_loss = defaultdict(float)

    for (idx, data) in enumerate(dataset):
        image = data["image"].unsqueeze(0).to(device).to(memory_format=torch.channels_last_3d)
        gt_meshes = [m.to(device) for m in data["meshes"]]
        gt_segmentation = data["segmentation"].unsqueeze(0).to(device)
        ground_truth = {"meshes": gt_meshes, "segmentation": gt_segmentation}
        batch_size = image.shape[0]

        batched_template = BatchTemplate.from_single_template(template, batch_size)
        batched_verts = batched_template.batch_vertex_coordinates()
        with torch.no_grad():
            predictions = net.predict(image, batched_verts, template_distance_map)

        batched_template.update_batched_vertices(predictions["deformed_vertices"], detach=False)
        predictions["meshes"] = batched_template.meshes_list

        loss_components = compute_loss_components(predictions, ground_truth, loss_evaluators, loss_config)

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
        template,
        loss_config,
        checkpointer,
        eval_frequency,
        point_cloud,
        template_distance_map
):
    assert len(dataloader) > 0
    assert len(validation_dataset) > 0
    assert 0 < eval_frequency <= 1
    if point_cloud is not None:
        assert isinstance(point_cloud, torch.Tensor)
        assert point_cloud.ndim == 2
        assert point_cloud.shape[-1] == 3

    eval_counter = 0
    eval_every = int(math.ceil(eval_frequency * len(dataloader)))

    norm_type = loss_config["norm_type"]
    print("USING NORM TYPE : ", norm_type)

    loss_evaluators = get_loss_evaluators(loss_config)

    running_training_loss = defaultdict(float)
    running_validation_loss = defaultdict(float)

    for (idx, data) in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)

        image = data["image"].to(device).to(memory_format=torch.channels_last_3d)
        gt_meshes = [m.to(device) for m in data["meshes"]]
        gt_segmentation = data["segmentation"].to(device)
        ground_truth = {"meshes": gt_meshes, "segmentation": gt_segmentation}
        batch_size = image.shape[0]

        batched_template = BatchTemplate.from_single_template(template, batch_size)
        batched_verts = batched_template.batch_vertex_coordinates()

        if point_cloud is not None:
            batched_point_cloud = point_cloud.repeat([batch_size,1,1]).to(device)
            batched_verts.append(batched_point_cloud)

        predictions = net.predict(image, batched_verts, template_distance_map)

        if point_cloud is not None:
            deformed_point_cloud = predictions["deformed_vertices"].pop()

        batched_template.update_batched_vertices(predictions["deformed_vertices"], detach=False)
        predictions["meshes"] = batched_template.meshes_list

        loss_components = compute_loss_components(predictions, ground_truth, loss_evaluators, loss_config)
        loss = loss_components["total"]
        loss.backward()
        optimizer.step()

        del predictions

        for (k, v) in loss_components.items():
            running_training_loss[k] += v.item()

        lr = optimizer.param_groups[0]["lr"]
        out_str = loss2str(loss_components)
        out_str = "\tBatch {:04d} | ".format(idx) + out_str + "LR {:1.3e}".format(lr)
        print(out_str)

        if (idx + 1) % eval_every == 0:
            print("\n\n\tEVALUATING MODEL")
            eval_counter += 1
            validation_loss_components = evaluate_model(
                net, 
                validation_dataset, 
                template, 
                template_distance_map, 
                loss_evaluators, 
                loss_config
            )
            out_str = "\t\t" + loss2str(validation_loss_components) + "\n\n"
            print(out_str)
            total_validation_loss = validation_loss_components["total"]
            save_data = {"model": net, "optimizer": optimizer}
            checkpointer.save_best_model(total_validation_loss, epoch, save_data)
            scheduler.step(total_validation_loss)

            for (k, v) in validation_loss_components.items():
                running_validation_loss[k] += v

    # save_data = {"model": net, "optimizer": optimizer}
    # checkpointer.save_checkpoint(epoch, save_data)

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
        template,
        loss_config,
        checkpointer,
        num_epochs,
        eval_frequency,
        point_cloud,
        template_distance_map
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
            template,
            loss_config,
            checkpointer,
            eval_frequency,
            point_cloud,
            template_distance_map
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
        checkpointer.save_loss(train_loss, "train_loss.csv")
        checkpointer.save_loss(validation_loss, "validation_loss.csv")

    return train_loss, validation_loss
