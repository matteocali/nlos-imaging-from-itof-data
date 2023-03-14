import torch
import torchvision.transforms as T
from utils.utils import normalize


class ItofNormalize(object):
    """
    Transformation class to normalize the iToF data dividing by the amplitude at 20MHz.
    """

    def __init__(self, n_freq: int):
        """
        Args:
            n_freq (int): number of frequencies used by the iToF sensor
        """

        self.n_frequencies = n_freq  # Number of frequencies used by the iToF sensor

    def __call__(self, sample: dict):
        """
        Args:
            sample (dict): dictionary containing the iToF data, the ground truth depth and the ground truth alpha
        Returns:
            dict: dictionary containing the normalized iToF data, the ground truth depth and the ground truth alpha
        """

        itof_data, gt_depth, gt_depth_cartesian, gt_mask = sample["itof_data"], sample["gt_depth"], sample["gt_depth_cartesian"], sample["gt_mask"]

        # Compute the normalization factor for the iToF data
        v_a = torch.sqrt(torch.square(itof_data[0, ...]) + torch.square(itof_data[self.n_frequencies, ...]))
        v_a = v_a.unsqueeze(0)
        
        # Scale the iToF raw data
        itof_data = itof_data / v_a

        return {"itof_data": itof_data, "gt_depth": gt_depth, "gt_depth_cartesian": gt_depth_cartesian, "gt_mask": gt_mask}


class ItofNormalizeWithAddLayer(object):
    """
    Transformation class to normalize the iToF data dividing by the amplitude at 20MHz.
    Other than that it also add at the beginning of the iToF data the amplitude at 20MHz normalized on its own.
    """

    def __init__(self, n_freq: int):
        """
        Args:
            n_freq (int): number of frequencies used by the iToF sensor
        """

        self.n_frequencies = n_freq  # Number of frequencies used by the iToF sensor

    def __call__(self, sample: dict):
        """
        Args:
            sample (dict): dictionary containing the iToF data, the ground truth depth and the ground truth alpha
        Returns:
            dict: dictionary containing the normalized iToF data, the ground truth depth and the ground truth alpha
        """

        itof_data, gt_depth, gt_depth_cartesian, gt_mask = sample["itof_data"], sample["gt_depth"], sample["gt_depth_cartesian"], sample["gt_mask"]
        
        # Compute the amplitude at 20MHz (normalization factor for the iToF data)
        ampl_20 = torch.sqrt(torch.square(itof_data[0, ...]) + torch.square(itof_data[self.n_frequencies, ...])).unsqueeze(0)
        
        # Scale the iToF raw data
        itof_data = itof_data / ampl_20
        
        # Add a dimension containing the amplitude at 20MHz rescaled by 10e9
        ampl_20 = ampl_20 / 10e9  # Rescale the amplitude at 20MHz
        itof_data = torch.cat((ampl_20, itof_data), dim=0)  # type: ignore

        return {"itof_data": itof_data, "gt_depth": gt_depth, "gt_depth_cartesian": gt_depth_cartesian, "gt_mask": gt_mask}


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
            - gt_depth_cartesian: ground truth depth in cartesian coordinates
            - gt_alpha: ground truth alpha
        """

        itof_data, gt_depth, gt_depth_cartesian, gt_mask = sample["itof_data"], sample["gt_depth"], sample["gt_depth_cartesian"], sample["gt_mask"]

        # Change the background value from bg_value to target_value (for the gt_depth)
        gt_depth = torch.where(gt_depth == self.bg_value, self.target_value, gt_depth)
        gt_depth_cartesian = torch.where(gt_depth_cartesian == self.bg_value, self.target_value, gt_depth_cartesian)

        return {"itof_data": itof_data, "gt_depth": gt_depth, "gt_depth_cartesian": gt_depth_cartesian, "gt_mask": gt_mask}


class RandomRotation(T.RandomRotation):
    """
    Transformation class to rotate the iToF data.
    If fill is set to float("inf"), the filling area will be set to the original value of the input image nont to a fixed value.
    """

    def __init__(self, degrees: int, interpolation: T.InterpolationMode = T.InterpolationMode.NEAREST, expand: bool = False, center = None, fill: float = 0):
        """
        Args:
            degrees (int): Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).
            interpolation (InterpolationMode): Desired interpolation enum defined by
                :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            expand (bool): Optional expansion flag. If true, expands the output
                to make it large enough to hold the entire rotated image. If false or omitted,
                make the output image the same size as the input image. Note that the expand flag assumes rotation around the center and no translation.
            center (2-tuple, optional): Optional center of rotation (x, y). Origin is the upper left corner.
                Default is the center of the image.
            fill (sequence or number, optional): Pixel fill value for area outside the rotated image. If int or float, the value is used for all bands respectively.
        """

        self.wrap = False
        if fill == float("inf"):
            fill = int(-1e10)
            self.wrap = True
        super().__init__(degrees, interpolation, expand, center, int(fill))
    

    def __call__(self, sample: torch.Tensor):
        """
        Args:
            sample (Tensor): Tensor image of size (C, H, W) to be rotated.
        Returns:
            Tensor: Rotated Tensor image.
        """

        m_sample = super(RandomRotation, self).__call__(sample)  # Call the super class to rotate the imnage

        if self.wrap:
            m_sample = torch.where(m_sample == -1e10, sample, m_sample)  # If the fill value is set to float("inf"), set the filling area to the original value of the input image

        return m_sample


class RandomAffine(T.RandomAffine):
    """
    Transformation class to rotate the iToF data.
    If fill is set to float("inf"), the filling area will be set to the original value of the input image nont to a fixed value.
    """

    def __init__(self, degrees: int, translate = None, scale = None, shear = None, interpolation: T.InterpolationMode = T.InterpolationMode.NEAREST, fill: float = 0, center = None):
        """
        Args:
            degrees (int): Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).
            translate (tuple, optional): tuple of maximum absolute fraction for horizontal
                and vertical translations. For example translate=(a, b), then horizontal shift
                is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is randomly sampled in the range
                -img_height * b < dy < img_height * b. Will not translate by default.
            scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
                randomly sampled from the range a <= scale <= b. Will keep original scale by default.
            shear (sequence or float, optional): Range of degrees to select from.
                If shear is a number, a shear parallel to the x axis in the range (-shear, +shear) will be apllied. Else if shear is a 2-tuple,
                a shear parallel to the x axis in the range (shear[0], shear[1]) will be applied. Else if shear is a 4-tuple,
                a shear parallel to the x axis in the range (shear[0], shear[1]) and a shear parallel to the y axis in the range (shear[2], shear[3]) will be applied.
            interpolation (InterpolationMode): Desired interpolation enum defined by
                :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            fill (sequence or number, optional): Pixel fill value for area outside the rotated image. If int or float, the value is used for all bands respectively.
            center (2-tuple, optional): Optional center of rotation (x, y). Origin is the upper left corner.
                Default is the center of the image.
        """

        self.wrap = False
        if fill == float("inf"):
            fill = int(-1e10)
            self.wrap = True
        super().__init__(degrees, translate, scale, shear, interpolation, int(fill), center)
    

    def __call__(self, sample: torch.Tensor):
        """
        Args:
            sample (Tensor): Tensor image of size (C, H, W) to be rotated.
        Returns:
            Tensor: Rotated Tensor image.
        """

        m_sample = super(RandomAffine, self).__call__(sample)  # Call the super to and transform the imnage

        if self.wrap:
            m_sample = torch.where(m_sample == -1e10, sample, m_sample)  # If the fill value is set to float("inf"), set the filling area to the original value of the input image

        return m_sample


class AddGaussianNoise(object):
    """Add gaussian noise to a tensor."""

    def __init__(self, mean: float = 0., std: float = 1.):
        """
        Args:
            mean (float): mean of the gaussian distribution
            std (float): standard deviation of the gaussian distribution
        """

        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """

        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        """
        Returns:
            str: string representation of the object
        """
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)