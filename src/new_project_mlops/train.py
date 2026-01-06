from pathlib import Path

import matplotlib.pyplot as plt
import torch

from new_project_mlops.data import load_corrupt_mnist_datasets
from new_project_mlops.model import MyAwesomeModel as Model



def train():
    model = Model()

    train_set, _ = load_corrupt_mnist_datasets(Path("data/raw"))
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    statistics = {"train_loss": [], "train_accuracy": []}

    for epoch in range(10):
        model.train()
        for i, (data, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (output.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = models_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")

    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(figures_dir / "training_statistics.png")


if __name__ == "__main__":
    train()
