import torch


class STECleanFunc(torch.autograd.Function):
    """
    Straight through estimator
    Clean the iToF data, everything below 0.05 is set to 0
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        param:
            - ctx: context
            - x: input tensor
        return:
            - output tensor
        """

        return torch.where(abs(x) < 0.05, 0, x)  # Clean the itof data
    
    @staticmethod
    def backward(ctx, grad_out) -> torch.Tensor:
        """
        Backward pass
        Estimate the gradient of the hard threshold using a sigmoid
        param:
            - ctx: context
            - grad_out: gradient output
        return:
            - output tensor
        """

        return grad_out
    

class STEThresholdFunc(torch.autograd.Function):
    """
    Straight through estimator
    Compute the hard threshold of the depth data
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Forward pass
        param:
            - ctx: context
            - x: input tensor
            - threshold: threshold value
        return:
            - output tensor
        """

        return torch.where(x <= threshold, 0, 1)
    
    @staticmethod
    def backward(ctx, grad_out) -> torch.Tensor:
        """
        Backward pass
        Estimate the gradient of the hard threshold using a sigmoid
        param:
            - ctx: context
            - grad_out: gradient output
        return:
            - output tensor
        """

        # return torch.nn.Sigmoid()(grad_out)
        return grad_out


class StraightThroughEstimator(torch.nn.Module):
    """Straight through estimator implementation"""

    def __init__(self, task: str, threshold: float | None = None) -> None:
        """
        Constructor
        param:
            - task: task to perform (clean or threshold)
            - threshold: threshold value
        """

        super().__init__()
        self.task = task

        if self.task == "threshold":
            self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        param:
            - x: input tensor
        return:
            - output tensor
        """

        if self.task == "clean":
            return STECleanFunc.apply(x)
        elif self.task == "threshold":
            return STEThresholdFunc.apply(x, self.threshold)
        else:
            raise ValueError(f"Unknown task {self.task}")
