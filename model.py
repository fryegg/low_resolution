from torchvision.models import resnet34
import torch.nn as nn

model = resnet34(pretrained=True)
model.fc = nn.Linear(512, 10)

ix = 0
for name, child in model.named_children():
    print(name, '-%d' %ix)
    ix += 1

model_list = list(model.modules())[1:]
print(model.modules)

