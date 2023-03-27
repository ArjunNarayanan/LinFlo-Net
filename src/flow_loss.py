from src.loss import *
from torch.nn import CrossEntropyLoss


def compute_loss_components(predictions, ground_truth, loss_evaluators, loss_config):
    loss_components = {}

    # chamfer distance and normal
    if loss_config.get("chamfer_distance", 0) > 0:
        chd, chn = average_chamfer_distance_between_meshes(predictions["meshes"],
                                                           ground_truth["meshes"],
                                                           loss_config["norm_type"])
        loss_components["chamfer_distance"] = loss_config["chamfer_distance"] * chd
        loss_components["chamfer_normal"] = loss_config["chamfer_normal"] * chn

    # Divergence integral
    if loss_config.get("divergence", 0) > 0:
        div_loss = loss_config["divergence"] * average_divergence_loss(predictions["divergence_integral"])
        loss_components["divergence"] = div_loss

    # Cross entropy
    if loss_config.get("cross_entropy", 0) > 0:
        ce_evaluator = loss_evaluators["cross_entropy"]
        ce_loss = loss_config["cross_entropy"] * ce_evaluator(predictions["segmentation"], ground_truth["segmentation"])
        loss_components["cross_entropy"] = ce_loss

    # Dice loss
    if loss_config.get("dice", 0) > 0:
        dice_evaluator = loss_evaluators["dice"]
        dice = loss_config["dice"] * dice_evaluator(predictions["segmentation"], ground_truth["segmentation"])
        loss_components["dice"] = dice

    # Edge loss
    if loss_config.get("edge", 0) > 0:
        edge_loss = loss_config["edge"] * average_mesh_edge_loss(predictions["meshes"])
        loss_components["edge"] = edge_loss

    # laplace
    if loss_config.get("laplace", 0) > 0:
        laplace_loss = loss_config["laplace"] * average_laplacian_smoothing_loss(predictions["meshes"])
        loss_components["laplace"] = laplace_loss

    if loss_config.get("normal", 0) > 0:
        normal_loss = loss_config["normal"] * average_normal_consistency_loss(predictions["meshes"])
        loss_components["normal"] = normal_loss

    total = sum(loss_components.values())
    loss_components["total"] = total

    return loss_components


def get_loss_evaluators(loss_config):
    evaluators = {}
    if "cross_entropy" in loss_config:
        evaluators["cross_entropy"] = CrossEntropyLoss(reduction="mean")
    if "dice" in loss_config:
        evaluators["dice"] = SoftDiceLoss()

    return evaluators
