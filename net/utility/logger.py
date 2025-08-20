import os
import argparse
import time
from typing import  Union

def log_operation(log_dir: str,
                  mode: str,
                  parser: argparse.Namespace,
                  extra: Union[dict, None] = None) -> None:
    """
    Append a human-readable line to <log_dir>/operations.log.

    :param log_dir: directory to log.
    :param mode: operation mode.
    :param parser: parser instance.
    :param extra: extra arguments.
    """
    os.makedirs(log_dir, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] MODE={mode.upper()}"
    if parser is not None:
        line += " | " + " ".join([f"{k}={v}" for k, v in vars(parser).items()])
    if extra:
        line += " | " + " ".join([f"{k}={v}" for k, v in extra.items()])
    with open(os.path.join(log_dir, "log.txt"), "a") as f:
        f.write(line + "\n")
