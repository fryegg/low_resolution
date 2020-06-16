from torchvision.models import resnet34

model = resnet34(pretrained=True)
print(list(model.modules()))