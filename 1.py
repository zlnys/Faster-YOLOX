import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
np.random.seed(1234)

# 这里用的是torchvision已经封装好的MINST数据集
trainset = torchvision.datasets.MNIST(
    root='MNIST',  # root是下载MNIST数据集保存的路径，可以自行修改
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

testset = torchvision.datasets.MNIST(
    root='MNIST',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

trainloader = DataLoader(dataset=trainset, batch_size=100,
                         shuffle=True)  # DataLoader是一个很好地能够帮助整理数据集的类，可以用来分批次，打乱以及多线程等操作
testloader = DataLoader(dataset=testset, batch_size=100, shuffle=True)

#可视化某一批数据
train_img,train_label=next(iter(trainloader))   #iter迭代器，可以用来便利trainloader里面每一个数据，这里只迭代一次来进行可视化
fig, axes = plt.subplots(10, 10, figsize=(10, 10))
axes_list = []
#输入到网络的图像
for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        axes[i, j].imshow(train_img[i*10+j,0,:,:],cmap="gray")    #这里画出来的就是我们想输入到网络里训练的图像，与之对应的标签用来进行最后分类结果损失函数的计算
        axes[i, j].axis("off")
#对应的标签
print(train_label)
class convolutio(nn.Module):
    def __init__(self, ks, ch_in, ch_out):
        super(convolutio, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=ks, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=ks, stride=1, padding=1 ,bias=True),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class CNN(nn.Module):
    def __init__(self, kernel_size, in_cl, out_cl):
        super(CNN, self).__init__()
        fecture = [16, 32, 64, 128, 256]
        self.conv1 = convolutio(kernel_size, in_cl, fecture[0])
        self.conv2 = convolutio(kernel_size, fecture[0], fecture[1])
        self.conv3 = convolutio(kernel_size, fecture[1], fecture[2])
        self.conv4 = convolutio(kernel_size, fecture[2], fecture[3])
        self.conv5 = convolutio(kernel_size, fecture[3], fecture[4])
        self.fc = nn.Sequential(
            nn.Linear(fecture[4] * 28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        device = x.device
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x5 = x5.view(x5.size()[0], -1)
        out = self.fc(x5)
        return out

#网络参数定义
  #此处根据电脑配置进行选择，如果没有cuda就用cpu
device = torch.device("cpu")
net = CNN(3,1,1).to(device = device,dtype = torch.float32)
epochs = 50  #训练轮次
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-8)  #使用Adam优化器
criterion = nn.CrossEntropyLoss()  #分类任务常用的交叉熵损失函数
train_loss = []

# Begin training
MinTrainLoss = 999
for epoch in range(1, epochs + 1):
    total_train_loss = []
    net.train()
    start = time.time()
    for input_img, label in trainloader:
        input_img = input_img.to(device=device, dtype=torch.float32)  # 我们同样地，需要将我们取出来的训练集数据进行torch能够运算的格式转换
        label = label.to(device=device, dtype=torch.float32)  # 输入和输出的格式都保持一致才能进行运算
        optimizer.zero_grad()  # 每一次算loss前需要将之前的梯度清零，这样才不会影响后面的更新
        pred_img = net(input_img)
        loss = criterion(pred_img, label.long())
        loss.backward()
        optimizer.step()
        total_train_loss.append(loss.item())
    train_loss.append(np.mean(total_train_loss))  # 将一个minibatch里面的损失取平均作为这一轮的loss
    end = time.time()
    # 打印当前的loss
    print(
        "epochs[%3d/%3d] current loss: %.5f, time: %.3f" % (epoch, epochs, train_loss[-1], (end - start)))  # 打印每一轮训练的结果

    if train_loss[-1] < MinTrainLoss:
        torch.save(net.state_dict(), "./model_min_train.pth")  # 保存loss最小的模型
        MinTrainLoss = train_loss[-1]
