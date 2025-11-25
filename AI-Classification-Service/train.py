import torch
import copy
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

class AIClassificationService:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent
        self.DATA_DIR = base_dir / "data"
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 20
        self.LEARNING_RATE = 0.001
        self.NUM_WORKERS = 4
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.DEVICE}")

    def _initialize_model(self):
        IMG_SIZE = 224
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        train_dir = self.DATA_DIR / "train"
        val_dir = self.DATA_DIR / "val"
        test_dir = self.DATA_DIR / "test"

        train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
        val_ds = datasets.ImageFolder(val_dir, transform=val_test_transform)
        test_ds = datasets.ImageFolder(test_dir, transform=val_test_transform)

        self.train_loader = DataLoader(train_ds, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)
        self.val_loader = DataLoader(val_ds, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=self.NUM_WORKERS)
        self.test_loader = DataLoader(test_ds, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=self.NUM_WORKERS)

        self.CLASS_NAMES = train_ds.classes
        self.NUM_CLASSES = len(self.CLASS_NAMES)
        print(f"Classes: {self.CLASS_NAMES}")

    def build_model(self, num_classes: int) -> nn.Module:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    
    def train_epoch(self, model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def evaluate(self, model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

def main():
    service = AIClassificationService()
    service._initialize_model()

    model = service.build_model(service.NUM_CLASSES).to(service.DEVICE)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_optimize, lr=service.LEARNING_RATE)

    best_val_acc = 0.0
    best_state_dict = None
    for epoch in range(1, service.NUM_EPOCHS + 1):
        train_loss, train_acc = service.train_epoch(model, service.train_loader, criterion, optimizer, service.DEVICE)
        val_loss, val_acc = service.evaluate(model, service.val_loader, criterion, service.DEVICE)

        print(f"Epoch {epoch:02d}/{service.NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = copy.deepcopy(model.state_dict())

    if best_state_dict is not None:
        print(f"Loading best model with Val Acc: {best_val_acc:.4f}")
        model.load_state_dict(best_state_dict)

    test_loss, test_acc = service.evaluate(model, service.test_loader, criterion, service.DEVICE)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    save_path = Path("models")
    save_path.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": service.CLASS_NAMES,
        },
        save_path / "resnet18_transfer.pth"
    )
    print(f"Model saved to {save_path / 'resnet18_transfer.pth'}")

if __name__ == "__main__":
    main()