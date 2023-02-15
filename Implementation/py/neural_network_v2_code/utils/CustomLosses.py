import torch

class BalancedMAELoss(torch.nn.Module):
    """
    Custom loss function for the neural network.
    This loss function is a weighted mean absolute error loss function.
    Given a mask will average the absolute error of the masked pixels and the bg.
    """

    def __init__(self, reduction: str = "none"):
        """
        Args:
            reduction (str): reduction method (mean, sum, weight_mean, none)
        """

        super().__init__()
        self.reduction = reduction
        if reduction == "weight_mean":
            reduction = "none"
        # Create the mean absolute error loss function
        self.mae = torch.nn.L1Loss(reduction=reduction)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): predicted value
            target (torch.Tensor): ground truth value
            mask (torch.Tensor): mask
        Returns:
            total_mae (torch.Tensor): total mean absolute error
        """
        
        # Calculate the mean absolute error
        mae = self.mae(pred, target)

        # Calculate the mean absolute error of the masked pixels
        if self.reduction == "weight_mean":
            obj_mae = mae * mask
            mean_obj_mae = torch.sum(obj_mae) / torch.sum(mask)
            bg_mae = mae * (1 - mask)
            mean_bg_mae = torch.sum(bg_mae) / torch.sum(1 - mask)
            mae = (mean_obj_mae + mean_bg_mae) / 2
        
        return mae


class BalancedBCELoss(torch.nn.Module):
    """
    Custom loss function for the neural network.
    This loss function is a binary cross entropy error loss function.
    Given a mask will average the absolute error of the masked pixels and the bg.
    """

    def __init__(self, reduction: str = "none"):
        """
        Args:
            reduction (str): reduction method (mean, sum, weight_mean, none)
        """

        super().__init__()
        self.reduction = reduction
        if reduction == "weight_mean":
            reduction = "none"
        # Create the mean absolute error loss function
        self.bce = torch.nn.BCELoss(reduction=reduction)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): predicted value
            target (torch.Tensor): ground truth value
            mask (torch.Tensor): mask
        Returns:
            total_mae (torch.Tensor): total mean absolute error
        """
        
        # Calculate the mean absolute error
        bce = self.bce(pred, target)

        # Calculate the mean absolute error of the masked pixels
        if self.reduction == "weight_mean":
            obj_bce = bce * mask
            mean_obj_bce = torch.sum(obj_bce) / torch.sum(mask)
            bg_bce = bce * (1 - mask)
            mean_bg_bce = torch.sum(bg_bce) / torch.sum(1 - mask)
            bce = (mean_obj_bce + mean_bg_bce) / 2
        
        return bce
