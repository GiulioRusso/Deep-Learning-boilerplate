import os
from typing import List, Any

from pandas import read_csv
from skimage import io
from torch.utils.data import Dataset

# TODO: adjust the dataset class based on your needs

class MyDataset(Dataset):
    """
    My Dataset class
    """

    def __init__(self,
                 images_dir: str,
                 annotations_dir: str,
                 filename_list: List[str],
                 transforms: Any):
        """
        __init__ method: run one when instantiating the object

        :param images_dir: images directory
        :param annotations_dir: annotations directory
        :param filename_list: filename list
        :param transforms: transforms dataset to apply
        """

        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.filename_list = filename_list
        self.transforms = transforms

    def __len__(self) -> int:
        """
        __len__ method: returns the number of samples in dataset
        :return: number of samples in dataset
        """

        return len(self.filename_list)

    def __getitem__(self,
                    idx: int) -> dict:
        """
        __getitem__ method: loads and return a sample from the dataset at given index

        :param idx: sample index
        :return: sample dictionary
        """

        # ----- #
        # IMAGE #
        # ----- #
        image_filename = self.filename_list[idx] + ".tif"
        image_path = os.path.join(self.images_dir, image_filename)
        image = io.imread(image_path)  # numpy.ndarray

        # ---------- #
        # ANNOTATION #
        # ---------- #
        annotation_filename = self.filename_list[idx] + ".csv"
        annotation_path = os.path.join(self.annotations_dir, annotation_filename)

        annotation_column = ["ANNOTATION"]
        annotation = read_csv(filepath_or_buffer=annotation_path, usecols=annotation_column).values  # numpy.ndarray

        # sample dict
        sample = {
            'filename': self.filename_list[idx],
            'image': image,
            'annotation': annotation,
        }

        # transforms
        if self.transforms:
            sample = self.transforms(sample)

        return sample
