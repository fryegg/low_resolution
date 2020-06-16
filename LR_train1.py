import torchvision.models as models
import torch.nn as nn
import os
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.models import resnet34
import argparse
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.01, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', default=128, help='')
parser.add_argument('--batch_size_test', default=100, help='')
parser.add_argument('--num_worker', default=4, help='')
parser.add_argument('--logdir', type=str, default='logs', help='')
parser.add_argument('--train_type', type=str, default='high', help='')
parser.add_argument('--low_size', type=int, default=16, help='')

parser.add_argument('--distillation', type=int, default=0, help='0: False, 1:True')
parser.add_argument('--gamma', type=float, default=0.5, help='')

args = parser.parse_args()

## GPU Setting
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

## Transformation (Augmentation, Resizing)
transform_high = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_low = transforms.Compose([
	transforms.Resize([args.low_size, args.low_size]),
	transforms.Resize([32, 32]),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

## Dataset and DataLoader
dataset_high_train = CIFAR10(root='../data', train=True,
							 download=False, transform=transform_high)
dataset_high_test = CIFAR10(root='../data', train=False,
							download=False, transform=transform_high)

train_high_loader = DataLoader(dataset_high_train, batch_size=args.batch_size,
							   shuffle=False)
test_high_loader = DataLoader(dataset_high_test, batch_size=args.batch_size_test,
							  shuffle=False)

dataset_low_train = CIFAR10(root='../data', train=True,
							 download=False, transform=transform_low)
dataset_low_test = CIFAR10(root='../data', train=False,
							download=False, transform=transform_low)

train_low_loader = DataLoader(dataset_low_train, batch_size=args.batch_size,
							   shuffle=False)
test_low_loader = DataLoader(dataset_low_test, batch_size=args.batch_size_test,
							  shuffle=False)


# there are 10 classes so the dataset name is cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer',
	       'dog', 'frog', 'horse', 'ship', 'truck')

## Model
if args.train_type == 'high':
	model = resnet34(pretrained=True)
	model.fc = nn.Linear(512, 10)
	model = nn.DataParallel(model).cuda()

else:
	model = resnet34(pretrained=True)
	model.fc = nn.Linear(512, 10)
	model = nn.DataParallel(model).cuda()

## Trainer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 1e-4)

def train(epoch, model,train_loader):
	model.train()

	train_loss = 0
	correct = 0
	total = 0

	for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
		inputs = inputs.cuda()
		targets = targets.cuda()
		outputs = model(inputs)
		loss = criterion(outputs, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

	acc = 100 * correct / total
	# print('train epoch : {} | loss: {:.3f} | acc: {:.3f}'.format(
	# 	epoch, train_loss / len(train_loader), acc))


def test(epoch, best_acc, model, test_loader, type='high'):
	model.eval()

	test_loss = 0
	correct = 0
	total = 0

	with torch.no_grad():
		for batch_idx, (i nputs, targets) in enumerate(tqdm(test_loader)):
			inputs = inputs.cuda()
			targets = targets.cuda()
			outputs = model(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

	acc = 100 * correct / total
	print('test epoch : {} | loss: {:.3f} | acc: {:.3f}'.format(
		epoch, test_loss / len(test_high_loader), acc))

	if best_acc < acc:
		best_acc = acc
		torch.save(model.module.state_dict(), './model/%s_best.pt' %type)

	return best_acc

def distill_train(epoch, gamma, high_model, low_model, train_high_loader, train_low_loader):
	low_model.train()
	high_model.eval()

	# No grad for high_model
	for param in high_model.parameters():
		param.requires_grad = False

	# Option
	train_loss = 0
	correct = 0
	total = 0

	class attention_func(object):
		def __init__(self, model):
			self.start = nn.Sequential(*[model.module.conv1, model.module.bn1, model.module.relu, model.module.maxpool])
			self.layer1 = model.module.layer1
			self.layer2 = model.module.layer2
			self.layer3 = model.module.layer3
			self.layer4 = model.module.layer4

		def forward(self, input):
			out0 = self.start(input)
			out1 = self.layer1(out0)
			out2 = self.layer2(out1)

			return torch.max(out0, dim=1)[0], torch.max(out1, dim=1)[0], torch.max(out2, dim=1)[0]

	attention_high = attention_func(high_model)
	attention_low = attention_func(low_model)

	mse_cri = nn.MSELoss()

	if (epoch == 4) and (gamma == 8):
		gamma = gamma / 2

	for (high, low) in zip(train_high_loader, train_low_loader):
		input_high, input_low = high[0].cuda(), low[0].cuda()
		target_high, target_low = high[1].cuda(), low[1].cuda()

		# Distillation Loss
		h0, h1, h2 = attention_high.forward(input_high)
		l0, l1, l2 = attention_low.forward(input_low)

		attention_loss = (0.7 * mse_cri(h0, l0) + 0.3 * mse_cri(h1, l1)) / 2.
		# Classification Loss
		outputs = low_model(input_low)
		soft_loss = criterion(outputs, target_low)

		# Total Loss
		loss = soft_loss + gamma * attention_loss


		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += target_low.size(0)
		correct += predicted.eq(target_low).sum().item()

	acc = 100 * correct / total
	# print('train epoch : {} | loss: {:.3f} | acc: {:.3f}'.format(
	# 	epoch, train_loss / len(train_low_loader), acc))


if __name__ == '__main__':
	# Option
	epoch_num = 100
	epoch = 0

	# High train model
	if args.train_type == 'high':
		best_acc = 0
		for epoch in range(epoch_num):
			train(epoch, model, train_high_loader)
			best_acc = test(epoch, best_acc, model, test_high_loader, type='high')

	# Low train model
	elif (args.train_type == 'low') and (args.distillation == 0):
		best_acc = 0
		for epoch in range(epoch_num):
			train(epoch, model, train_low_loader)
			best_acc = test(epoch, best_acc, model, test_low_loader, type='low_%d' %args.low_size)

	# Distillation model (High -> Low)
	else:
		best_acc = 0
		# Load the pre-trained model
		high_pretrain = torch.load('./model/high_best.pt')
		high_model = resnet34(pretrained=True)
		high_model.fc = nn.Linear(512, 10)
		high_model.load_state_dict(high_pretrain)
		high_model = nn.DataParallel(high_model).cuda()

		# Train with distillation
		for epoch in range(epoch_num):
			distill_train(epoch, args.gamma, high_model, model, train_high_loader, train_low_loader)
			best_acc = test(epoch, best_acc, model, test_low_loader, type='distill_%d' %args.low_size)
