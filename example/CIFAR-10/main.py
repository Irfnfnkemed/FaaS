import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from models import *


import sys
import os

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_path not in sys.path:
    sys.path.append(project_path)




from FaaS.intra.elements import FaaSDataLoader
from FaaS.intra.job import IntraOptim
import FaaS.intra.env as env


# 定义参数
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--epochs', default=60, type=int, help='number of epochs')
parser.add_argument('--model', default='ResNet18', type=str, help='model')
args = parser.parse_args()



# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# 训练函数
def train(job: IntraOptim, loss_func, epoch):
    job.model.train()
    device = env.rank()
    running_loss = 0.0
    correct = 0
    total = 0
    job.beg_epoch()
    for batch_idx, (data, target) in enumerate(train_loader):
        with job.sync_or_not():
            data, target = data.to(device), target.to(device)
            job.optimizer.zero_grad()
            output = job.model(data)
            loss = loss_func(output, target)
            loss.backward()
            job.optimizer.step()      
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            if batch_idx % 25 == 24:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {running_loss / 100:.6f}')
                running_loss = 0.0
            if job.is_break:
                break
    print(f'Accuracy: {100 * correct / total}%')
  

# 测试函数
def test(job: IntraOptim, loss_func):
    job.model.eval()
    device = env.rank()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = job.model.module(data)
            test_loss += loss_func(output, target).item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        print(f'Test set: Average loss: {test_loss / len(test_loader.dataset):.4f}, Accuracy: {100 * correct / total}%')                       



if __name__ == '__main__':
    # 设备配置
    device = env.rank()
    
    # 数据加载
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = FaaSDataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # 模型定义，这里以 ResNet18 为例
    model = eval(args.model)()
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    job_optim = IntraOptim(model, train_loader, test_loader, optimizer, 1)

    # 训练和测试循环
    for epoch in range(1, args.epochs + 1):   
        train(job_optim, criterion, epoch)
        test(job_optim, criterion)