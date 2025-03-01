from pandas import read_csv


def read_split(path_split_case: str) -> dict:
    """
    Read data split

    :param path_split_case: path split
    :return: split dictionary
    """

    # TODO: adjust based on your split file structure

    # read csv
    dtype_mapping = {"FILENAME": str}  # define data type for cols
    data_split_header = ["INDEX", "FILENAME", "SPLIT"]
    data_split = read_csv(filepath_or_buffer=path_split_case, usecols=data_split_header, dtype=dtype_mapping).values

    index = data_split[:, 0]
    filename = data_split[:, 1]
    split = data_split[:, 2]

    split_dict = {
        'index': index.tolist(),
        'case': filename.tolist(),
        'split': split.tolist()
    }

    return split_dict
