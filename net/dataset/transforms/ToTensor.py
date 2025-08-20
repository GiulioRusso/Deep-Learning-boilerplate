import torch


class ToTensor(object):
    """
    ToTensor: convert nd arrays to tensor
    """

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        image, image_mask, vessel_mask, annotation = sample['image'], sample['image_mask'], sample['vessel_mask'], sample['annotation']

        image = torch.from_numpy(image).float()
        annotation = torch.from_numpy(annotation).float()

        # TODO: Adjust the sample based on your dataset class

        sample = {
            'filename': sample['filename'],
            'image': image,
            'annotation': annotation
        }

        return sample
