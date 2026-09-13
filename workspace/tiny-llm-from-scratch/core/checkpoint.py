import torch


def save_checkpoint(path, payload):

    torch.save(payload, path)


def load_checkpoint(path):

    return torch.load(
        path,
        map_location="cpu"
    )