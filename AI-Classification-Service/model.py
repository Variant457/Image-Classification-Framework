import torch
import io
from torchvision import models, transforms
from torch import nn
from pathlib import Path
from PIL import Image

class AIClassificationModel:
    def __init__(self, device: torch.device | None = None):
        baseDir = Path(__file__).resolve().parent
        checkpoint_path = baseDir / "models" / "resnet18_transfer.pth"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.class_names = checkpoint["class_names"]
        num_classes = len(self.class_names)

        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model
        IMG_SIZE = 224
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def prepare_tensor(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0) # type: ignore
        return tensor.to(self.device)
    
    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        tensor = self.prepare_tensor(image)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        label = self.class_names[idx.item()]

        return {
            "class": label,
            "confidence": float(conf.item()),
        }
    
    def predictFromBytes(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes))
        return self.predict(image)
    
def main():
    classifier = AIClassificationModel()
    img = Image.new("RGB", (224, 244), color="white")
    result = classifier.predict(img)
    print("Dummy Prediction:", result)

if __name__ == "__main__":
    main()