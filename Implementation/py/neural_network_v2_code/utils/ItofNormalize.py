import torch


class ItofNormalize(object):
    """
    Normalize the iToF data with the amplitude at 20MHz
    param:
        - n_freq: number of frequencies used by the iToF sensor
    return:
        - itof_data: normalized iToF data
        - gt_depth: normalized ground truth depth
        - gt_alpha: normalized ground truth alpha
    """

    def __init__(self, n_freq: int):
        self.n_frequencies = n_freq  # Number of frequencies used by the iToF sensor

    def __call__(self, sample: dict):
        itof_data, gt_depth, gt_alpha = sample["itof_data"], sample["gt_depth"], sample["gt_alpha"]

        v_a = torch.sqrt(torch.square(itof_data[..., 0]) + torch.square(itof_data[..., self.n_frequencies]))
        v_a = torch.unsqueeze(v_a, dim=-1)
        
        # Scale the iToF raw data
        itof_data /= v_a

        return {"itof_data": itof_data, "gt_depth": gt_depth, "gt_alpha": gt_alpha}