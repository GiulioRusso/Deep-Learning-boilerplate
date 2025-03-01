import numpy as np


class Padding(object):
    """
    Padding: apply padding to image to match specified dimensions.
    """

    def __init__(self, target_height: int = 512,  target_width: int = 512):
        """
        Initialize with target dimensions.

        :param target_height: Desired height of the image after padding.
        :param target_width: Desired width of the image after padding.
        """
        self.target_height = target_height
        self.target_width = target_width

    def __call__(self, sample: dict) -> dict:
        """
        Apply padding to the image and mask in the sample to reach the target dimensions.

        :param sample: Sample dictionary containing 'image', 'mask', and 'annotation'.
        :return: Modified sample with padded image and mask.
        """
        image = sample['image']
        mask = sample['mask']
        annotation = sample['annotation']

        # calculate padding amounts
        height, width = image.shape[:2]
        pad_height = (self.target_height - height) // 2 if height < self.target_height else 0
        pad_width = (self.target_width - width) // 2 if width < self.target_width else 0

        pad_width_tuple = (pad_width, self.target_width - width - pad_width)
        pad_height_tuple = (pad_height, self.target_height - height - pad_height)

        # apply padding
        image_pad = np.pad(image, (pad_height_tuple, pad_width_tuple, (0, 0)), mode='constant', constant_values=0)
        mask_pad = np.pad(mask, (pad_height_tuple, pad_width_tuple), mode='constant', constant_values=0)

        # adjust annotations if necessary
        annotation_pad = annotation.copy()
        annotation_pad[:, 0] += pad_width  # Adjust x coordinates
        annotation_pad[:, 1] += pad_height  # Adjust y coordinates

        padded_sample = {
            'filename': sample['filename'],
            'image': image_pad,
            'mask': mask_pad,
            'annotation': annotation_pad,
        }

        return padded_sample
