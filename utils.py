import csv
import os
import random
from enum import IntEnum

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

torch.manual_seed(1004)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomSketch:

    def __init__(self, p=1.0, invert_p=0.25):
        self.p = p
        self.invert_p = invert_p

    def __call__(self, img):
        if random.random() >= self.p:
            return img

        gray = img.convert('L')
        style = random.choice(['edge', 'edge', 'pencil', 'silhouette'])

        if style == 'edge':
            if HAS_CV2:
                arr = np.asarray(gray)
                low = random.randint(30, 70)
                high = random.randint(120, 180)
                edges = cv2.Canny(arr, low, high)
                k = random.choice([1, 2, 2, 3])
                if k > 1:
                    edges = cv2.dilate(edges, np.ones((k, k), np.uint8))
                sketch = Image.fromarray(255 - edges)  
            else:
                edges = gray.filter(ImageFilter.FIND_EDGES)
                edges = ImageOps.autocontrast(edges)
                sketch = ImageOps.invert(edges)

        elif style == 'pencil':
            inv = ImageOps.invert(gray)
            blur = inv.filter(ImageFilter.GaussianBlur(radius=random.uniform(8, 16)))
            g = np.asarray(gray, dtype=np.float32)
            b = np.asarray(blur, dtype=np.float32)
            dodge = g * 255.0 / (255.0 - b + 1e-3)
            dodge = np.clip(dodge, 0, 255).astype(np.uint8)
            sketch = Image.fromarray(dodge)

        else:  
            if HAS_CV2:
                arr = np.asarray(gray)
                _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                if (binary == 0).mean() > 0.5:
                    binary = 255 - binary
                
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                sketch = Image.fromarray(binary)
            else:
                
                arr = np.asarray(gray)
                t = int(arr.mean())
                binary = np.where(arr < t, 0, 255).astype(np.uint8)
                if (binary == 0).mean() > 0.5:
                    binary = 255 - binary
                sketch = Image.fromarray(binary)

        
        if random.random() < self.invert_p:
            sketch = ImageOps.invert(sketch.convert('L'))

        return sketch.convert('RGB')


photo_transform = transforms.Compose([
    transforms.RandomResizedCrop(320, scale=(0.65, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomRotation(10, fill=255),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.2),
])


sketch_always_transform = transforms.Compose([
    transforms.RandomResizedCrop(320, scale=(0.65, 1.0)),
    transforms.RandomHorizontalFlip(),
    RandomSketch(p=1.0, invert_p=0.25),
    transforms.RandomRotation(10, fill=255),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


train_transform = photo_transform


eval_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_gray_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

custom_transform = eval_transform


def make_data_loader(args):
    photo_ds = datasets.ImageFolder(args.data, transform=photo_transform)
    sketch_ds = datasets.ImageFolder(args.data, transform=sketch_always_transform)
    train_dataset = ConcatDataset([photo_ds, sketch_ds])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=getattr(args, 'num_workers', 2),
        pin_memory=True,
        drop_last=False,
    )

    return train_loader


class SketchClass(IntEnum):
    ANT = 0
    BANANA = 1
    BEE = 2
    CANDLE = 3
    CANNON = 4
    CASTLE = 5
    CHURCH = 6
    CUP = 7
    GEYSER = 8
    HAMMER = 9


CLASS_NAMES = [class_name.name.lower() for class_name in SketchClass]
CLASS_TO_IDX = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}


def format_duration(seconds):
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def acc(pred, label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()


def save_training_plot(train_losses, train_accuracies, save_path):
    if len(train_losses) != len(train_accuracies):
        raise ValueError('train_losses and train_accuracies must have the same length')
    if not train_losses:
        raise ValueError('training history is empty')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, train_losses, marker='o')
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_accuracies, marker='o')
    axes[1].set_title('Train Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def save_prediction_csv(save_path, filenames, preds, labels, classes=None):
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'pred_index', 'pred_class', 'true_index', 'true_class', 'correct']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for filename, pred, label in zip(filenames, preds, labels):
            has_label = label >= 0
            writer.writerow({
                'filename': filename,
                'pred_index': pred,
                'pred_class': classes[pred] if classes and pred < len(classes) else '',
                'true_index': label if has_label else '',
                'true_class': classes[label] if has_label and classes and label < len(classes) else '',
                'correct': int(pred == label) if has_label else '',
            })