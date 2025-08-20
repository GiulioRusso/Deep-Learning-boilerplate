import os
from typing import List, Optional


def _list_versions(exp_dir: str) -> List[str]:
    """
    Returns a list of all versions of the experiment directory.
    """
    if not os.path.isdir(exp_dir):
        return []
    return sorted([d for d in os.listdir(exp_dir) if d.startswith("version_") and os.path.isdir(os.path.join(exp_dir, d))])

def get_latest_version(exp_dir: str) -> Optional[str]:
    """
    Returns the latest version of the experiment directory.
    """
    vs = _list_versions(exp_dir)
    if not vs:
        return None

    # pick the numerically largest version_* suffix if possible
    def _vnum(v):
        try:
            return int(v.split("_", 1)[1])
        except Exception:
            return -1

    return sorted(vs, key=_vnum)[-1]