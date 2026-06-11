import argparse
import csv
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from model import BaseModel
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from utils import CLASS_NAMES, CLASS_TO_IDX, save_prediction_csv, eval_transform, eval_gray_transform


class ImageDataset(Dataset):

    def __init__(self, root_dir, transform=None, manifest_path=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        if manifest_path and os.path.isfile(manifest_path):
            label_map = class_to_idx or {}
            with open(manifest_path, newline='', encoding='utf-8-sig') as f:
                self.samples = [
                    (row['filename'], label_map.get(row.get('class'), -1))
                    for row in csv.DictReader(f)
                    if row.get('filename')
                ]
        else:
            self.samples = [
                (name, -1) for name in sorted(os.listdir(root_dir))
                if os.path.isfile(os.path.join(root_dir, name))
                and name.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_name)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(img_path)
        img = Image.open(img_path).convert('RGB')

        # TTA: 같은 이미지의 두 가지 view(원본 정규화 / grayscale 정규화)를 반환
        data = eval_transform(img)
        data_gray = eval_gray_transform(img)
        return data, data_gray, label, img_name


def inference(args, data_loader, model):
    """ model inference with TTA (original + grayscale view 평균) """

    model.eval()
    preds = []
    labels = []
    filenames = []

    with torch.no_grad():
        pbar = tqdm(data_loader)
        for i, (x, x_gray, y, name) in enumerate(pbar):

            image = x.to(args.device)
            image_gray = x_gray.to(args.device)

            # 두 view의 softmax 확률을 평균 (후처리: 스케치 이미지에 강건)
            prob = F.softmax(model(image), dim=1) + F.softmax(model(image_gray), dim=1)

            _, predicted = torch.max(prob, 1)
            preds.extend(map(lambda t: t.item(), predicted))
            labels.extend(map(lambda t: t.item(), y))
            filenames.extend(name)

    return filenames, preds, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2026 Computer Vision Assignment 2')
    parser.add_argument('--load-model', default='checkpoints/model.pth', help="Model's state_dict")
    parser.add_argument('--batch-size', default=16, type=int, help='test loader batch size')
    parser.add_argument('--dataset', default='data/test_images', help='image dataset directory')
    parser.add_argument('--csv-output', default='results/results.csv', help='detailed prediction csv file')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    num_classes = 10
    classes = CLASS_NAMES

    # torchvision model
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    model.load_state_dict(torch.load(args.load_model, map_location=device))
    model.to(device)

    # load dataset in test image folder
    manifest_path = os.path.join(args.dataset, 'test_manifest.csv')
    test_data = ImageDataset(
        args.dataset,
        manifest_path=manifest_path,
        class_to_idx=CLASS_TO_IDX,
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    # write model inference
    filenames, preds, labels = inference(args, test_loader, model)
    save_prediction_csv(args.csv_output, filenames, preds, labels, classes)
    print(f"Detailed results saved to: {args.csv_output}")

    if labels and all(label >= 0 for label in labels):
        accuracy = sum(int(pred == label) for pred, label in zip(preds, labels)) / len(labels)
        print("Test Accuracy : {:.5f}".format(accuracy))
