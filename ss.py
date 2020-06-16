from torchvision.models import resnet34
import torch.nn as nn

model = resnet34(pretrained=True)
model.fc = nn.Linear(512, 10)


model_list = list(model.modules())[1:]