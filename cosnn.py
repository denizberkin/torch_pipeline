import os

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torchvision import datasets, transforms
from tqdm import tqdm

from models.models import LinearModel, LowRankModel


def load_mnist_data(bs: int = 16):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST(root="./data/datasets", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data/datasets", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader


def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    num_epochs: int,
    device: torch.device,
) -> tuple:
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        avg_losses = []
        running_loss = 0.0
        all_labels = []
        all_preds = []

        with tqdm(train_loader, desc="Batches", leave=False) as t:
            for _, (images, labels) in enumerate(t):
                images = images.view(-1, 28 * 28).to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.detach().cpu().numpy())
                all_preds.extend(preds.detach().cpu().numpy())

                t.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        avg_losses.append(avg_loss)
        accuracy = (torch.tensor(all_labels) == torch.tensor(all_preds)).float().mean().item() * 100
        f1 = f1_score(all_labels, all_preds, average="weighted")
        tqdm.write(
            f"Epoch {epoch+1}/{num_epochs}, \
                   Loss: {avg_loss:.4f}, \
                   Accuracy: {accuracy:.2f}%, \
                   F1 Score: {f1:.4f}"
        )

    return model, avg_losses


def test_model(model, test_loader, device: torch.device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        all_labels = []
        all_preds = []

        for images, labels in tqdm(test_loader, desc="Testing Batches"):
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")
    tqdm.write(f"Test Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}")
    return accuracy, f1, all_labels, all_preds


def main():
    LR = 3.0e-4
    BS = 256  # 64k sample her epoch, rtx3060
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    train_loader, test_loader = load_mnist_data(bs=BS)
    model = LowRankModel(in_features=28 * 28, hidden_size=[128, 64], out_features=10, rank=4).to(DEVICE)
    linear_model = LinearModel(in_features=28 * 28, hidden_size=[128, 64], out_features=10).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    linear_optimizer = torch.optim.Adam(linear_model.parameters(), lr=LR)

    print(sum(p.numel() for p in model.parameters()))
    print(sum(p.numel() for p in linear_model.parameters()))

    print("Training Low Rank Model...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=10, device=DEVICE)
    print("Testing Linear Model...")
    train_model(
        linear_model,
        train_loader,
        criterion,
        linear_optimizer,
        num_epochs=10,
        device=DEVICE,
    )
    if not os.path.exists("models/"):
        os.makedirs("models")
    print("Saving models...")
    torch.save(model.state_dict(), os.path.join("models", "lowrank_model_10.pth"))
    torch.save(linear_model.state_dict(), os.path.join("models", "linear_model_10.pth"))

    print("Testing Low Rank Model...")
    test_model(model, test_loader, device=DEVICE)
    print("Testing Linear Model...")
    test_model(linear_model, test_loader, device=DEVICE)


if __name__ == "__main__":
    main()
