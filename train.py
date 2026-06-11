import argparse
import os
import time

import numpy as np
from tqdm import tqdm
from utils import acc, make_data_loader, save_training_plot
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

import torch
import torch.nn as nn


def train(args, data_loader, model):
    # Label smoothing: 작은 데이터셋(900장)에서 과적합/과신 완화
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # 차등 learning rate:
    #  - 사전학습 backbone은 낮은 lr로 미세조정 (사전학습 feature 보존)
    #  - 새로 초기화된 fc는 높은 lr로 빠르게 학습
    backbone_params = [p for name, p in model.named_parameters() if not name.startswith('classifier')]
    head_params = [p for name, p in model.named_parameters() if name.startswith('classifier')]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.learning_rate},
        {'params': head_params, 'lr': args.learning_rate * 10},
    ], weight_decay=1e-4)

    # Cosine annealing: 후반부 lr을 줄여 안정적으로 수렴
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {'loss': [], 'accuracy': []}

    for epoch in range(args.epochs):
        train_losses = []
        train_acc = 0.0
        total = 0

        model.train()
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False, dynamic_ncols=True)
        for i, (x, y) in enumerate(pbar):
            image = x.to(args.device)
            label = y.to(args.device)
            optimizer.zero_grad()

            output = model(image)

            label = label.view(-1)  # squeeze 대신 view(-1): batch=1일 때도 안전
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            total += label.size(0)

            train_acc += acc(output, label)
            pbar.set_postfix(
                loss=f"{np.mean(train_losses):.4f}",
                acc=f"{(train_acc / total) * 100:.2f}%",
            )

        scheduler.step()

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc / total * 100
        history['loss'].append(epoch_train_loss)
        history['accuracy'].append(epoch_train_acc)

        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {epoch_train_loss:.6f} - acc: {epoch_train_acc:.3f}%")
        torch.save(model.state_dict(), f'{args.save_path}/model.pth')
        save_training_plot(history['loss'], history['accuracy'], args.plot_output)

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2026 Computer Vision Assignment 2')
    parser.add_argument('--save-path', default='checkpoints/', help="Model's state_dict")
    parser.add_argument('--data', default='data/train_images', type=str, help='data folder')
    parser.add_argument('--plot-output', default='results/training_metrics.png', help='training metric plot path')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    num_classes = 10

    # hyperparameters
    args.epochs = 50
    args.learning_rate = 1e-4   # backbone용 낮은 lr (fc는 내부에서 x10)
    args.batch_size = 32
    args.num_workers = 2

    # check settings
    print("==============================")
    print("Save path:", args.save_path)
    print("Data:", args.data)
    print('Using Device:', device)
    print('Number of usable GPUs:', torch.cuda.device_count())

    # Print Hyperparameter
    print("Batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("Epochs:", args.epochs)
    print("==============================")

    # Make Data loader and Model
    train_loader = make_data_loader(args)

    # torchvision model (사전학습 ResNet50 전이학습)
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

    # change num_classes to 10
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    model.to(device)

    os.makedirs(args.save_path, exist_ok=True)
    start = time.perf_counter()
    train(args, train_loader, model)
    print(f"Training time: {time.perf_counter() - start:.2f} sec")
