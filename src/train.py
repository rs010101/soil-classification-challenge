import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import os
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse


def get_data_loaders(data_dir, batch_size=32, input_size=224):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Assuming grayscale, adjust for RGB
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes


def build_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_model(model, train_loader, val_loader, device, num_epochs=5, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

        evaluate(model, val_loader, device, split="Validation")

    return model


def evaluate(model, loader, device, split="Test"):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    acc = accuracy_score(targets, preds)
    print(f"{split} Accuracy: {acc:.4f}")
    return acc


def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path='model.pth', device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, classes = get_data_loaders(
        args.data_dir, batch_size=args.batch_size, input_size=args.input_size
    )

    model = build_model(num_classes=len(classes))

    if args.train:
        model = train_model(model, train_loader, val_loader, device,
                            num_epochs=args.epochs, lr=args.lr)
        save_model(model, args.model_path)

    if args.infer:
        model = load_model(model, args.model_path, device)
        evaluate(model, test_loader, device, split="Test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data/', help='Path to dataset root folder')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to save/load the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--infer', action='store_true', help='Run inference on test set')

    args = parser.parse_args()
    main(args)
