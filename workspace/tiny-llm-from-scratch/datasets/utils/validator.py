from pathlib import Path


def check_file(path):

    if not Path(path).exists():

        raise FileNotFoundError(path)

    return True