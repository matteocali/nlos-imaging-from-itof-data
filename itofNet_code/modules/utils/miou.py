import torch
from torchmetrics.functional.classification.jaccard import binary_jaccard_index


def mean_intersection_over_union(
    pred: torch.Tensor, target: torch.Tensor, bg_class_value: int
) -> torch.Tensor:
    """
    Function used to compute the binary mean Intersection over Union (mIoU)
        param:
            - pred: predicted data
            - target: target data
            - bg_class_value: background class value
        return:
            - mean intersection over union
    """

    iou_1 = binary_jaccard_index(
        torch.where(pred > bg_class_value, 1, 0),
        torch.where(target > bg_class_value, 1, 0),
    )
    iou_2 = binary_jaccard_index(
        torch.where(pred == bg_class_value, 1, 0),
        torch.where(target == bg_class_value, 1, 0),
    )

    return (iou_1 + iou_2) / 2
