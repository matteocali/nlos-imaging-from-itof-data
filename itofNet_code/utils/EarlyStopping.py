import numpy as np
from pathlib import Path
from torch import save, nn


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, tollerance: int = 5, min_delta: float = 0, save_path: Path = None, net: nn.Module = None):  # type: ignore
        """
        Args:
            tollerance (int, optional): How long to wait after last time validation loss improved. Defaults to 5.
            min_delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.
            save_path (Path, optional): Path where to save the model. Defaults to None.
            net (torch.nn.Module, optional): Model to save. Defaults to None.
        """

        self.tollerance = tollerance
        self.min_delta = min_delta
        self.save_path = save_path
        self.net = net
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.min_val_loss = np.Inf

    def __call__(self, validation_loss: float) -> tuple[float, bool]:
        """
        Args:
            validation_loss (float): Current validation loss
        Returns:
            tuple[float, bool]: Tuple containing the minimum validation loss and a boolean value indicating if the training should be stopped
        """

        if validation_loss < self.min_val_loss:
            self.min_val_loss = validation_loss
            self.counter = 0
            # Save the model if required
            if self.save_path is not None and self.net is not None:
                save(self.net.state_dict(), self.save_path)
        elif validation_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.tollerance:
                return float(self.min_val_loss), True
        return float(self.min_val_loss), False
