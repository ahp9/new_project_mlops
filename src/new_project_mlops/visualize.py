from pathlib import Path

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import typer

from new_project_mlops.model import MyAwesomeModel as Model


def get_train_dataset(processed_data_path: Path):
    train_images = torch.load(processed_data_path / "train_images.pt")
    train_target = torch.load(processed_data_path / "train_target.pt")
    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    return train_set


def visualize(model_checkpoint: str, figure_name: str = "model.png"):
    model = Model()
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    model.fc = torch.nn.Identity()

    train_loader = torch.utils.data.DataLoader(get_train_dataset(Path("data/processed")), batch_size=64, shuffle=False)

    embeddings, targets = [], []
    with torch.inference_mode():
        for images, target in train_loader:
            images = images
            feats = model(images)  # shape: [batch, feature_dim]
            embeddings.append(feats.cpu())  # move back to CPU for sklearn
            targets.append(target)

    embeddings = torch.cat(embeddings).numpy()
    targets = torch.cat(targets).numpy()

    # 5) t-SNE to 2D
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    # 6) Plot + save
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 10))
    for digit in range(10):
        mask = targets == digit
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=6, label=str(digit))
    plt.legend()
    plt.title("t-SNE of CNN features (train set)")
    plt.savefig(out_dir / figure_name, dpi=200)
    plt.close()

    print(f"Saved figure to: {out_dir / figure_name}")


if __name__ == "__main__":
    typer.run(visualize)
