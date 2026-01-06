from pathlib import Path

import torch
import typer

from new_project_mlops.data import load_corrupt_mnist_datasets
from new_project_mlops.model import MyAwesomeModel as Model


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    model = Model()
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = load_corrupt_mnist_datasets(Path("data/raw"))
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img, target
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)
