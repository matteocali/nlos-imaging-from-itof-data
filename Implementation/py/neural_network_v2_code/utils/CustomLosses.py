import torch


class BalancedMAELoss(torch.nn.Module):
    """
    Custom loss function for the neural network.
    This loss function is a weighted mean absolute error loss function.
    Given a mask will average the absolute error of the masked pixels and the bg.
    """

    def __init__(self, reduction: str = "none", pos_weight: torch.Tensor or None = None):  # type: ignore
        """
        Args:
            reduction (str): reduction method (mean, sum, weight_mean, only_gt, none)
            pos_weight (torch.Tensor or None): weight to be applied to pixel corresponding to the positive class
        """

        super().__init__()
        self.reduction = reduction
        if pos_weight is not None:
            self.pos_weight = pos_weight
            reduction = "none"
        elif reduction == "weight_mean" or reduction == "only_gt":
            reduction = "none"
        # Create the mean absolute error loss function
        self.mae = torch.nn.L1Loss(reduction=reduction)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:  # type: ignore
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
        if self.pos_weight is not None:
            if mask is None:
                mask = torch.where(target != 0, 1, 0)
            weighted_mask = torch.where(mask == 1, self.pos_weight, 1)
            mae = mae * weighted_mask
            mae = torch.mean(mae)
        elif self.reduction == "weight_mean":
            obj_mae = mae * mask
            mean_obj_mae = torch.sum(obj_mae) / torch.sum(mask)
            bg_mae = mae * (1 - mask)
            mean_bg_mae = torch.sum(bg_mae) / torch.sum(1 - mask)
            mae = (mean_obj_mae + mean_bg_mae) / 2
        elif self.reduction == "only_gt":
            mae = mae * mask
            mae = torch.sum(mae) / torch.sum(mask)
        
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
            reduction (str): reduction method (mean, sum, weight_mean, dual_weight_mean, none)
        """

        super().__init__()
        self.reduction = reduction
        if reduction == "weight_mean" or reduction == "dual_weight_mean":
            reduction = "none"
        # Create the mean absolute error loss function
        self.bce = torch.nn.BCELoss(reduction=reduction)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:  # type: ignore
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

        if mask is None:
            mask = target

        # Calculate the mean absolute error of the masked pixels
        if self.reduction == "weight_mean":
            # Compute the various partial masks
            obj_mask = mask
            bg_mask = 1 - mask

            # Compute the various partial masks
            obj_bce = bce * obj_mask
            mean_obj_bce = torch.sum(obj_bce) / torch.sum(obj_mask)
            bg_bce = bce * bg_mask
            mean_bg_bce = torch.sum(bg_bce) / torch.sum(bg_mask)

            # Compute the final loss
            bce = (mean_obj_bce + mean_bg_bce) / 2
        elif self.reduction == "dual_weight_mean":
            # Compute the various partial masks
            obj_mask = mask
            bg_mask = 1 - mask
            hard_pred = torch.where(pred > 0.5, 1, 0)
            border_mask = hard_pred - mask
            no_border = True if border_mask.all() == obj_mask.all() else False

            # Compute the various partial losses
            obj_bce = bce * obj_mask                                              # loss for the object
            mean_obj_bce = torch.sum(obj_bce) / torch.sum(obj_mask)               # mean loss for the object
            bg_bce = bce * bg_mask                                                # loss for the background
            mean_bg_bce = torch.sum(bg_bce) / torch.sum(bg_mask)                  # mean loss for the background
            if not no_border:
                border_bce = bce * border_mask                                    # loss for the border
                mean_border_bce = torch.sum(border_bce) / torch.sum(border_mask)  # mean loss for the border

                # Compute the final loss
                bce = (mean_obj_bce + mean_border_bce + mean_bg_bce) / 3
            else:
                # Compute the final loss
                bce = (mean_obj_bce + mean_bg_bce) / 2
        
        return bce
