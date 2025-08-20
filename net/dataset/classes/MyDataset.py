from typing import List, Any, Optional, Dict
from pathlib import Path

import pandas as pd
from skimage import io
from torch.utils.data import Dataset


# TODO: Adjust the dataset class based on your data type and organization

class MyDataset(Dataset):
    """
    Computer Vision Dataset class for images and annotations

    Expected data structure:
    data/
    ├── images/              # Directory containing image files
    │   ├── sample001.tif
    │   ├── sample002.tif
    │   └── ...
    ├── annotations/         # Directory containing annotation CSV files
    │   ├── sample001.csv
    │   ├── sample002.csv
    │   └── ...
    └── filenames.txt        # Text file with list of filenames (without extensions)

    Each annotation CSV should contain:
    - ANNOTATION column with target values
    - Additional columns as needed

    Usage:
    >>> filenames = ['sample001', 'sample002', 'sample003']  # read list from filenames.txt
    >>> dataset = MyDataset('data/images', 'data/annotations', filenames, transforms=None)
    """

    # constants
    DEFAULT_IMAGE_EXTENSION = ".tif"
    DEFAULT_ANNOTATION_EXTENSION = ".csv"
    DEFAULT_ANNOTATION_COLUMNS = "LABEL"

    def __init__(self,
                 images_dir: str,
                 annotations_dir: str,
                 filename_list: List[str],
                 transforms: Optional[Any] = None,
                 image_extension: str = DEFAULT_IMAGE_EXTENSION,
                 annotation_extension: str = DEFAULT_ANNOTATION_EXTENSION,
                 annotation_columns: Optional[List[str]] = DEFAULT_ANNOTATION_COLUMNS):
        """
        Initialize the dataset

        :param images_dir: directory containing image files
        :param annotations_dir: directory containing annotation files
        :param filename_list: list of filenames without extensions
        :param transforms: optional transforms to apply to samples
        :param image_extension: file extension for images
        :param annotation_extension: file extension for annotations
        :param annotation_columns: specific columns to load from annotations
        """
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.filename_list = filename_list
        self.transforms = transforms
        self.image_extension = image_extension
        self.annotation_extension = annotation_extension
        self.annotation_columns = annotation_columns

    def __len__(self) -> int:
        """
        Return the number of samples in dataset

        :return: number of samples
        """
        return len(self.filename_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and return a sample from the dataset at given index

        :param idx: sample index
        :return: dictionary containing sample data
        """
        # ensure valid index
        if idx >= len(self.filename_list):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.filename_list)}")
        filename = self.filename_list[idx]

        # > load image
        image_path = self.images_dir / f"{filename}{self.image_extension}"
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = io.imread(str(image_path))

        # > load annotation
        annotation_path = self.annotations_dir / f"{filename}{self.annotation_extension}"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        df = pd.read_csv(annotation_path, usecols=self.annotation_columns)
        annotation = df.values

        # create sample dictionary
        sample = {
            'filename': filename,
            'image': image,
            'annotation': annotation,
        }

        # apply transforms if provided
        if self.transforms:
            sample = self.transforms(sample)

        return sample

