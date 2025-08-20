import argparse
from typing import Tuple

from torchvision import transforms
from torchvision.transforms import Compose

from net.dataset.transforms.Add3ChannelsImage import Add3ChannelsImage
from net.dataset.transforms.MinMaxNormalization import MinMaxNormalization
from net.dataset.transforms.StandardNormalization import StandardNormalization
from net.dataset.transforms.ToTensor import ToTensor
from net.initialization.utility.get_yaml import get_yaml

# TODO: Define the needed transforms based on your needs

def dataset_transforms(normalization: str,
                       parser: argparse.Namespace,
                       statistics_path: str) -> Tuple[Compose, Compose, Compose]:
    """
    Collect dataset transforms for cropped-area view

    :param normalization: normalization type
    :param parser: parser of parameters-parsing
    :param statistics_path: statistics path to the YAML configuration file
    :return: train transforms,
             validation transforms,
             test transforms
    """

    # read statistics
    statistics = get_yaml(yaml_path=statistics_path)

    # ---- #
    # NONE #
    # ---- #
    if normalization == 'none':

        # train dataset transforms
        train_transforms = transforms.Compose([
            # DATA PREPARATION
            Add3ChannelsImage(),  # Add 3 Channels to image [C, H, W]
            ToTensor()  # To Tensor
        ])

        # validation dataset transforms
        val_transforms = transforms.Compose([
            # DATA PREPARATION
            Add3ChannelsImage(),  # Add 3 Channels to image [C, H, W]
            ToTensor()  # To Tensor
        ])

        # test dataset transforms
        test_transforms = transforms.Compose([
            # DATA PREPARATION
            Add3ChannelsImage(),  # Add 3 Channels to image [C, H, W]
            ToTensor()  # To Tensor
        ])

    # ------- #
    # MIN-MAX #
    # ------- #
    elif normalization == 'min-max':

        # train dataset transforms
        train_transforms = transforms.Compose([
            # DATA PREPARATION
            Add3ChannelsImage(),  # Add 3 Channels to image [C, H, W]
            ToTensor(),  # To Tensor
            # MIN-MAX NORMALIZATION
            MinMaxNormalization(min=statistics['min_max']['train']['min'], max=statistics['min_max']['train']['max'])  # min-max normalization
        ])

        # validation dataset transforms
        val_transforms = transforms.Compose([
            # DATA PREPARATION
            Add3ChannelsImage(),  # Add 3 Channels to image [C, H, W]
            ToTensor(),  # To Tensor
            # MIN-MAX NORMALIZATION
            MinMaxNormalization(min=statistics['min_max']['val']['min'], max=statistics['min_max']['val']['max'])  # min-max normalization
        ])

        # test dataset transforms
        test_transforms = transforms.Compose([
            # DATA PREPARATION
            Add3ChannelsImage(),  # Add 3 Channels to image [C, H, W]
            ToTensor(),  # To Tensor
            # MIN-MAX NORMALIZATION
            MinMaxNormalization(min=statistics['min_max']['test']['min'], max=statistics['min_max']['test']['max'])  # min-max normalization
        ])

    # --- #
    # STD #
    # --- #
    elif normalization == 'std':

        # train dataset transforms
        train_transforms = transforms.Compose([
            # DATA PREPARATION
            Add3ChannelsImage(),  # Add 3 Channels to image [C, H, W]
            ToTensor(),  # To Tensor
            # STANDARD NORMALIZATION
            StandardNormalization(mean=statistics['std']['train']['mean'], std=statistics['std']['train']['std'])  # standard normalization
        ])

        # validation dataset transforms
        val_transforms = transforms.Compose([
            # DATA PREPARATION
            Add3ChannelsImage(),  # Add 3 Channels to image [C, H, W]
            ToTensor(),  # To Tensor
            # STANDARD NORMALIZATION
            StandardNormalization(mean=statistics['std']['val']['mean'], std=statistics['std']['val']['std'])  # standard normalization
        ])

        # test dataset transforms
        test_transforms = transforms.Compose([
            # DATA PREPARATION
            Add3ChannelsImage(),  # Add 3 Channels to image [C, H, W]
            ToTensor(),  # To Tensor
            # STANDARD NORMALIZATION
            StandardNormalization(mean=statistics['std']['test']['mean'], std=statistics['std']['test']['std'])  # standard normalization
        ])

    else:
        raise ValueError(f"Unknown normalization type in {__file__}. Got {normalization}")

    return train_transforms, val_transforms, test_transforms