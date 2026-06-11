import csv
import os
import random
from enum import IntEnum

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Keep train/test split deterministic for comparable accuracy runs.
torch.manual_seed(1004)

# ImageNet 사전학습 모델용 정규화 상수 (사전학습 가중치를 제대로 활용하려면 필수)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomSketch:
    """실사 이미지를 확률적으로 '스케치 풍'으로 변환하는 증강.

    test 데이터에 스케치(흰 배경 + 검은 선) 이미지가 섞여 있으므로,
    train(실사)에서 스케치 도메인을 흉내 내어 도메인 갭을 줄인다.
    두 가지 스타일을 랜덤하게 사용:
      1) edge: 엣지맵을 반전 (검은 선 / 흰 배경)
      2) pencil: dodge blend 기반 연필 스케치 효과
    """

    def __init__(self, p=0.35):
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img

        gray = img.convert('L')

        if random.random() < 0.5:
            # 1) 엣지 기반: 윤곽선만 남기고 반전 -> 흰 배경에 검은 선
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edges = ImageOps.autocontrast(edges)
            sketch = ImageOps.invert(edges)
        else:
            # 2) 연필 스케치(dodge blend): gray / blur(invert(gray))
            inv = ImageOps.invert(gray)
            blur = inv.filter(ImageFilter.GaussianBlur(radius=random.uniform(8, 16)))
            g = np.asarray(gray, dtype=np.float32)
            b = np.asarray(blur, dtype=np.float32)
            dodge = g * 255.0 / (255.0 - b + 1e-3)
            dodge = np.clip(dodge, 0, 255).astype(np.uint8)
            sketch = Image.fromarray(dodge)

        return sketch.convert('RGB')


# 학습용 transform: 스케치 모사 + 색 제거 계열 증강 + ImageNet 정규화
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
    transforms.RandomHorizontalFlip(),
    RandomSketch(p=0.5),
    transforms.RandomGrayscale(p=0.25),          # 색 의존도 차단
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomRotation(10, fill=255),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.2),
])

# 평가/추론용 transform: 증강 없이 정규화만
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# 추론 TTA용: grayscale 버전 (스케치-유사 view)
eval_gray_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# 하위 호환 (기존 skeleton 코드가 import하는 이름)
custom_transform = eval_transform


def make_data_loader(args):
    # Get Dataset
    train_dataset = datasets.ImageFolder(args.data, transform=train_transform)

    # Get Dataloader
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
