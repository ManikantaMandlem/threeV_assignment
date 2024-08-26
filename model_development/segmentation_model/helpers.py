import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np


def compute_iou(gt_mask, pred_mask):
    # Ensure the masks are binary
    assert gt_mask.shape == pred_mask.shape, "Masks must have the same shape"
    # Compute the intersection and union
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    # Compute IoU
    iou = intersection / union if union != 0 else 0.0
    return iou


def image_resize(tensor, im_size, mode="nearest"):
    resized_tensor = F.interpolate(tensor, size=im_size, mode=mode)
    return resized_tensor


def visualize_batch(sample):
    sample["image"] = sample["image"].cpu().numpy()
    sample["mask"] = sample["mask"].cpu().numpy()
    # check if there is a pred mask as well to visualize
    pred_flag = False
    if "pred" in sample:
        sample["pred"] = sample["pred"].cpu().numpy()
        pred_flag = True
    images = sample["image"]
    masks = sample["mask"]
    if pred_flag:
        predictions = sample["pred"]
    batch_size = images.shape[0]
    # display three images at once if there is pred_flag
    if pred_flag:
        for i in range(batch_size):
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(images[i].squeeze(), cmap="gray")
            ax[0].set_title("Image")
            ax[1].imshow(masks[i].squeeze(), cmap="gray")  # Squeeze for single channel
            ax[1].set_title("Mask")
            ax[2].imshow(
                predictions[i].squeeze(), cmap="gray"
            )  # Squeeze for single channel
            ax[2].set_title("Prediction")
            plt.show()
    # just display image and gt mask
    else:
        for i in range(batch_size):
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(images[i].squeeze(), cmap="gray")
            ax[0].set_title("Image")
            ax[1].imshow(masks[i].squeeze(), cmap="gray")  # Squeeze for single channel
            ax[1].set_title("Mask")
            plt.show()


def log_clearml_metrics(metrics, clearml_task, epoch):
    for key, val in metrics.items():
        clearml_task.get_logger().report_scalar(
            title=key, series=key, iteration=epoch, value=val
        )


def safe_collate(batch):
    # make sure there are no None's in the batch created
    bs = len(batch)
    batch = list(filter(lambda x: x is not None, batch))
    if bs > len(batch):  # if there were None's, replace them with the existing data
        diff = bs - len(batch)
        batch = batch + batch[:diff]
    return torch.utils.data.dataloader.default_collate(batch)
