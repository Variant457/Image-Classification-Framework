import numpy as np
from pathlib import Path
from torchvision.datasets import CIFAR10

DATA_ROOT = Path("data")
SELECTED_CLASSES = ["cat", "dog"]
TRAIN_RATIO = 0.8
RANDOM_SEED = 56

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def main():
    cifar_train = CIFAR10(root="cifar_data", train=True, download=True)
    cifar_test = CIFAR10(root="cifar_data", train=False, download=True)

    print(f"Train set size: {len(cifar_train)}")
    print(f"Test set size: {len(cifar_test)}")
    print(f"Classes: {cifar_train.classes}")

    class_to_idx = cifar_train.class_to_idx
    selected_label_ids = [class_to_idx[name] for name in SELECTED_CLASSES]
    rng = np.random.default_rng(RANDOM_SEED)

    train_indices = {cls_id: [] for cls_id in selected_label_ids}
    for idx in range(len(cifar_train)):
        _, label = cifar_train[idx]
        if label in selected_label_ids:
            train_indices[label].append(idx)

    final_train_indices, final_val_indices = [], []
    for cls_id, idxs in train_indices.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        split_idx = int(len(idxs) * TRAIN_RATIO)
        final_train_indices.extend(idxs[:split_idx])
        final_val_indices.extend(idxs[split_idx:])

    test_indices = []
    for idx in range(len(cifar_test)):
        _, label = cifar_test[idx]
        if label in selected_label_ids:
            test_indices.append(idx)

    def dump_split(dataset, indices, split_name):
        for idx in indices:
            img, label = dataset[idx]
            class_name = dataset.classes[label]
            if class_name not in SELECTED_CLASSES:
                continue
            save_dir = DATA_ROOT / split_name / class_name
            ensure_dir(save_dir)
            img_path = save_dir / f"{split_name}_{class_name}_{idx:05d}.png"
            img.save(img_path)

    print("Saving training images...")
    dump_split(cifar_train, final_train_indices, "train")
    print("Saving validation images...")
    dump_split(cifar_train, final_val_indices, "val")
    print("Saving test images...")
    dump_split(cifar_test, test_indices, "test")
    print("Done! Dataset written under:", DATA_ROOT.resolve())

if __name__ == "__main__":
    main()