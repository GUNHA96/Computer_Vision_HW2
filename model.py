import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, num_classes=10, image_size=224):
        super(BaseModel, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(3 * image_size * image_size, 256)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        return self.layer2(x)
        
