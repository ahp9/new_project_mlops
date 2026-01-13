from pathlib import Path

import matplotlib.pyplot as plt
import torch
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from new_project_mlops.data import load_corrupt_mnist_datasets
from new_project_mlops.model import MyAwesomeModel as Model

wandb.login()


def train(lr: float = 0.001, batch_size: int = 64, epochs: int = 10):
    print("Training")
    print(f"Learning rate: {lr}, batch size: {batch_size}, epochs: {epochs}")

    run = wandb.init(
        project="new-project-mlops", config={"learning_rate": lr, "batch_size": batch_size, "epochs": epochs}
    )
    model = Model()

    train_set, _ = load_corrupt_mnist_datasets(Path("data/raw"))
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    preds, targets = [], []
    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (output.argmax(dim=1) == target).float().mean().item()

            preds.append(output.argmax(dim=1).detach().cpu())
            targets.append(target.detach().cpu())

            statistics["train_accuracy"].append(accuracy)
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = models_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    final_accuracy = accuracy_score(targets.numpy(), preds.numpy())
    final_precision = precision_score(targets.numpy(), preds.numpy(), average="weighted")
    final_recall = recall_score(targets.numpy(), preds.numpy(), average="weighted")
    final_f1 = f1_score(targets.numpy(), preds.numpy(), average="weighted")

    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={
            "accuracy": final_accuracy,
            "precision": final_precision,
            "recall": final_recall,
            "f1": final_f1,
        },
    )
    artifact.add_file(str(checkpoint_path))
    run.log_artifact(artifact)

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
