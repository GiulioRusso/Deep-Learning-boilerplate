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
        image_mask = torch.from_numpy(image_mask.copy()).float()
        vessel_mask = torch.from_numpy(vessel_mask.copy()).float()
        annotation = torch.from_numpy(annotation).float()

        sample = {'filename': sample['filename'],
                  'image': image,
                  'image_mask': image_mask,
                  'vessel_mask': vessel_mask,
                  'annotation': annotation,
                  }

        return sample
