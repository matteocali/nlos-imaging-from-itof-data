import torch


class ItofNormalize(object):
    """
    Transformation class to normalize the iToF data with the amplitude at 20MHz
    """

    def __init__(self, n_freq: int):
        """
        Default constructor
        param:
            - n_freq: number of frequencies used by the iToF sensor
        """

        self.n_frequencies = n_freq  # Number of frequencies used by the iToF sensor

    def __call__(self, sample: dict):
        """
        Function to normalize the iToF data with the amplitude at 20MHz
        param:
            - sample: dictionary containing the iToF data, the ground truth depth and the ground truth alpha
        return:
            - itof_data: normalized iToF data
            - gt_depth: ground truth depth
            - gt_alpha: ground truth alpha
        """

        itof_data, gt_depth, gt_mask = sample["itof_data"], sample["gt_depth"], sample["gt_mask"]

        v_a = torch.sqrt(torch.square(itof_data[..., 0]) + torch.square(itof_data[..., self.n_frequencies]))
        v_a = torch.unsqueeze(v_a, dim=-1)
        
        # Scale the iToF raw data
        itof_data /= v_a

        return {"itof_data": itof_data, "gt_depth": gt_depth, "gt_mask": gt_mask}


class ChangeBgValue(object):
    """
    Transformation class to change the background value to a specific value
    """

    def __init__(self, bg_value: int, target_value: int):
        """
        Default constructor
        param:
            - bg_value: background value
            - target_value: target value
        """

        self.bg_value = bg_value          # Background value
        self.target_value = target_value  # Target value

    def __call__(self, sample: dict):
        """
        Function to change the background value to a specific value
        param:
            - sample: dictionary containing the iToF data, the ground truth depth and the ground truth alpha
        return:
            - itof_data: iToF data
            - gt_depth: ground truth depth
            - gt_alpha: ground truth alpha
        """

        itof_data, gt_depth, gt_mask = sample["itof_data"], sample["gt_depth"], sample["gt_mask"]

        # Change the background value from bg_value to target_value (for the gt_depth)
        gt_depth = torch.where(gt_depth == self.bg_value, self.target_value, gt_depth)

        return {"itof_data": itof_data, "gt_depth": gt_depth, "gt_mask": gt_mask}