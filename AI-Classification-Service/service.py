import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

class AIClassificationService:
    def __init__(self):
        self.DATA_DIR = Path("data")
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 10
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

def main():
    service = AIClassificationService()
    service._initialize_model()

if __name__ == "__main__":
    main()