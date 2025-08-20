from typing import Tuple, List
import csv
from torch.utils.data import Dataset, Subset


def dataset_split(data_split: str,
                  dataset: Dataset) -> Tuple[Subset, Subset, Subset]:
    """
    Read a CSV split file and create three :class:`torch.utils.data.Subset` objects
    (train, validation, test) by selecting rows based on the `SPLIT` column.

    The CSV is expected to have **three columns** (case-insensitive):
      - `INDEX`: 0-based integer index into `dataset`.
      - `FILENAME`: optional filename or identifier (not used by this function, but
        kept for traceability and future checks).
      - `SPLIT`: one of `train`, `val` (or `validation`), or `test`.

    Expected split file (CSV):

        INDEX,FILENAME,SPLIT
        0,img_000.png,train
        1,img_001.png,train
        2,img_002.png,val
        3,img_003.png,test
        4,img_004.png,train
        5,img_005.png,val
        6,img_006.png,test

    Important:
    The order of rows in the split file **must match exactly** the order of samples
    in the :class:`Dataset` passed to this function.
    That is, `INDEX` corresponds to the position of the sample in `dataset[i]`.
    If the dataset changes before the call of this split function (e.g., shuffling, filtering, or new files),
    the split file must be regenerated to ensure correct alignment.

    :param data_split: data split path
    :param dataset: dataset
    :return: dataset train,
             dataset validation,
             dataset test
    """

    # TODO: Adjust the split logic based on your needs

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    n_items = len(dataset)

    with open(data_split, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        # normalize header names to upper-case for robustness
        if reader.fieldnames is None:
            raise ValueError("CSV must have a header row with INDEX,FILENAME,SPLIT.")
        header_map = {name.upper(): name for name in reader.fieldnames}
        required = {"INDEX", "SPLIT"}
        if not required.issubset(header_map):
            raise ValueError(f"CSV is missing required columns: {sorted(required - set(header_map))}")

        col_index = header_map["INDEX"]
        col_split = header_map["SPLIT"]

        for row_num, row in enumerate(reader, start=2):  # start=2 (1-based + header)
            raw_idx = (row.get(col_index) or "").strip()
            raw_split = (row.get(col_split) or "").strip().lower()

            # parse index
            try:
                idx = int(raw_idx)
            except ValueError as e:
                raise ValueError(f"Row {row_num}: INDEX '{raw_idx}' is not an integer.") from e

            if not (0 <= idx < n_items):
                raise ValueError(
                    f"Row {row_num}: INDEX {idx} out of bounds for dataset of length {n_items}."
                )

            # map split label
            if raw_split in ("train",):
                train_idx.append(idx)
            elif raw_split in ("val", "validation"):
                val_idx.append(idx)
            elif raw_split in ("test",):
                test_idx.append(idx)
            else:
                raise ValueError(
                    f"Row {row_num}: unknown SPLIT '{row.get(col_split)}'. "
                    "Expected one of: train, val/validation, test."
                )

    # build subsets
    dataset_train = Subset(dataset=dataset, indices=train_idx)
    dataset_val = Subset(dataset=dataset, indices=val_idx)
    dataset_test = Subset(dataset=dataset, indices=test_idx)

    return dataset_train, dataset_val, dataset_test
