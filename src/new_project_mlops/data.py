from pathlib import Path

import torch
from torch.utils.data import Dataset

import typer


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder.mkdir(parents=True, exist_ok=True)

        # Load raw data
        train_set, test_set = load_corrupt_mnist_datasets(self.data_path)

        train_images = train_set.tensors[0]
        train_target = train_set.tensors[1]
        test_images = test_set.tensors[0]
        test_target = test_set.tensors[1]

        train_images = normalize(train_images)
        test_images = normalize(test_images)
        torch.save(train_images, output_folder / "train_images.pt")
        torch.save(train_target, output_folder / "train_target.pt")
        torch.save(test_images, output_folder / "test_images.pt")
        torch.save(test_target, output_folder / "test_target.pt")


def normalize(images):
    """Normalize a tensor of images by subtracting the mean and dividing by the standard deviation."""
    return (images - images.mean()) / images.std()


def load_corrupt_mnist_datasets(data_path: Path):
    """
    Load the corrupt MNIST datasets.

    Args:
        data_path (Path): Path to the data folder.

    Returns:
        Tuple[TensorDataset, TensorDataset]: Train and test datasets.
    """
    train_images = []
    train_target = []

    for i in range(6):
        train_images.append(torch.load(data_path / f"train_images_{i}.pt"))
        train_target.append(torch.load(data_path / f"train_target_{i}.pt"))

    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images = torch.load(data_path / "test_images.pt")
    test_target = torch.load(data_path / "test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
