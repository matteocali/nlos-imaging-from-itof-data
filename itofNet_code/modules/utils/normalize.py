import numpy as np
import torch


def normalize(
    data: np.ndarray or torch.Tensor, bounds: dict[str, dict[str, float]]
) -> np.ndarray or torch.Tensor:
    """
    Function used to normalize the data
        param:
            - data: data to be normalized
            - bounds: bounds of the data
                - actual: actual bounds of the data
                    - lower: lower bound of the data
                    - upper: upper bound of the data
                - desired: desired bounds of the data
                    - lower: lower bound of the data
                    - upper: upper bound of the data
        return:
            - normalized data
    """

    return bounds["desired"]["lower"] + (data - bounds["actual"]["lower"]) * (
        bounds["desired"]["upper"] - bounds["desired"]["lower"]
    ) / (bounds["actual"]["upper"] - bounds["actual"]["lower"])
