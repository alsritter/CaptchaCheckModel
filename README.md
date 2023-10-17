---
title: 如何使用 CNN 做个验证码识别
date: 2023-10-13 15:44:27
tags: []
categories: []
updated: 2023-10-13 15:44:27
cover: https://image.alsritter.icu/img/202203022250694.jpg
---

## 生成数据集
因为只是一个试验，所以这里找个比较简单的验证码生成库

```bash
go get github.com/afocus/captcha
```

检查当前安装的字体

```bash
# Linux 可以使用 fc-list
dir C:\Windows\Fonts
```

编写一个生成数据集的函数

```go
package main

import (
	"fmt"
	"image/color"
	"image/png"
	"os"

	"github.com/afocus/captcha"
)

func main() {
	// 创建一个验证码对象
	cap := captcha.New()

	// 设置验证码的字体
	if err := cap.SetFont("MSYH.TTC"); err != nil {
		panic(err.Error())
	}

	// 设置验证码的大小
	cap.SetSize(128, 64)

	// 设置验证码的干扰强度
	cap.SetDisturbance(captcha.MEDIUM)

	// 设置验证码的前景色
	cap.SetFrontColor(color.RGBA{255, 255, 255, 255})

	// 设置验证码的背景色
	cap.SetBkgColor(
		color.RGBA{255, 0, 0, 255},    // 红色
		color.RGBA{0, 255, 0, 255},    // 绿色
		color.RGBA{0, 0, 255, 255},    // 蓝色
		color.RGBA{255, 255, 0, 255},  // 黄色
		color.RGBA{255, 0, 255, 255},  // 紫色
		color.RGBA{0, 255, 255, 255},  // 青色
		color.RGBA{255, 165, 0, 255},  // 橙色
		color.RGBA{128, 128, 128, 255} // 灰色
	)

	dirName := "captcha"

	// 检查目录是否存在
	if _, err := os.Stat(dirName); os.IsNotExist(err) {
		// 如果目录不存在，则创建它
		errDir := os.MkdirAll(dirName, 0755)
		if errDir != nil {
			fmt.Println()
			panic(fmt.Sprintf("error creating directory: %v", errDir))
		}
	}

	// 生成1000张验证码图片
	for i := 0; i < 1000; i++ {
		img, str := cap.Create(4, captcha.ALL)
		fileName := fmt.Sprintf("./"+dirName+"/%d_%s.png", i, str)
		file, err := os.Create(fileName)
		if err != nil {
			panic(err.Error())
		}

		png.Encode(file, img)
		file.Close()
	}
}
```

生成出来的验证码

![](https://image.alsritter.icu/img202310131636897.png)

## 引入依赖
```py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
```


## 数据的预处理
上面的验证码不是正方形的，为了数据集中的图像预处理成正方形，可以先执行填充操作，然后再进行缩放。填充可以确保图像的长和宽相等，而不改变图像的内容。

这里需要得到一个 128x128 的正方形图像。：

1. 首先，使用`transforms.Resize`，但只对图像的较短的一边进行缩放，保持其宽高比。
2. 使用`transforms.Pad`来填充较长的一边，使其与较短的一边相等，从而得到一个正方形图像。
3. 最后，再次使用`transforms.Resize`将其调整到所需的尺寸。

代码示例如下：

```python
from torchvision import transforms

def pad_to_square(img):
    # 获取图像的尺寸
    w, h = img.size
    # 计算要填充的最大边
    max_dim = max(w, h)
    # 计算左右和上下需要填充的尺寸
    pad_w = (max_dim - w) // 2
    pad_h = (max_dim - h) // 2
    # 返回填充后的图像
    return transforms.Pad((pad_w, pad_h))(img)

transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),  # 转换为RGB格式
    transforms.Lambda(pad_to_square),                # 将图像填充为正方形
    transforms.Resize((128, 128)),                   # 将图像调整为所需尺寸
    transforms.ToTensor(),
])
```

此方法首先将图像填充为正方形，然后再调整其大小。这样，图像内容不会因为不同的缩放比例而变形，从而可以确保图像的内容保持一致性。

## 定义数据集类
因为上面的验证码是自己生成的数据集，所以需要自己编写数据集采集类，这里使用 PyTorch 的 `torch.utils.data.Dataset` 类，需要实现 `__len__` 和 `__getitem__` 方法

```py
# 定义分类
nums = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


# 创建字符到整数的映射（字典推导）
char_to_int = {char: i for i, char in enumerate(nums)}

class CaptchaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(
            root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        label_str = self.image_files[idx].split('_')[1].split('.')[
            0]  # 提取验证码的名称作为 Label
        label = torch.tensor([char_to_int[char]
                             for char in label_str], dtype=torch.long)  # 转换为整数张量（就是取得当前这个 char 在字符串的索引）
        if self.transform:
            image = self.transform(image)
        # label: 是一个一维张量，维度为 [sequence_length]，其中 sequence_length 是字符序列的长度。每个元素是一个类别标签（索引）。
        return image, label

# 训练数据集
train_dataset = CaptchaDataset(root_dir='captcha', transform=transform)
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True)

# 测试数据集
test_dataset = CaptchaDataset(root_dir='test_captcha', transform=transform)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=True)
```

## 定义相关函数
下面会定义训练函数，评测函数，以及绘制函数

### 定义损失函数和优化器

首先是定义损失函数和优化器

```py
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
# 定义StepLR调度器，每10个epoch，学习率乘以gamma=0.9
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
```

### 绘制每一轮的收敛
定义一个绘制每一轮收敛情况的函数

```py
def update_plots(losses, accuracies):
    """实时更新并显示损失和准确率的图表"""

    # 清除先前的图表
    plt.clf()

    fig, axs = plt.subplots(2)

    # 绘制损失曲线
    axs[0].plot(losses, 'r', label='Training Loss')
    axs[0].set_title('Training Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # 绘制准确率曲线
    axs[1].plot(accuracies, 'g', label='Training Accuracy')
    axs[1].set_title('Training Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].legend()

    # 使用plt.pause以实时更新图表
    plt.pause(0.001)

    # 保持图表开放
    plt.show(block=False)
```

### 定义训练函数
然后定义训练的函数，就是一个很标准的训练过程

```py
def train_model(model, trainloader, criterion, optimizer, epochs, device):
    train_losses = []
    train_accuracies = []  # 用于记录每个周期的准确率
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0  # 记录这个周期中正确预测的字符数量
        total = 0    # 记录这个周期中总的字符数量

        for _, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # outputs 的维度是 [batch_size, sequence_length, num_chars]
            # labels 的维度是 [batch_size, sequence_length]
            # outputs[:, i, :]：此操作从 outputs 张量中选取第 i 个字符的预测，对于批次中的所有样本。
            #                   其维度为 [batch_size, num_chars]，表示每个样本的第 i 个字符的类别预测分数。
            # labels[:, i]：此操作从 labels 张量中选取第 i 个字符的实际类别标签，对于批次中的所有样本。
            #               其维度为 [batch_size]，表示每个样本的第 i 个字符的实际类别。
            #
            # 通过遍历 sequence_length，我们可以为每个字符位置计算损失。
            loss = sum([criterion(outputs[:, i, :], labels[:, i])
                       for i in range(labels.size(1))])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # 计算这个批次的准确率
            _, predicted = torch.max(outputs.data, 2)
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()
        
				# 每个epoch结束后，调用scheduler的step方法来更新学习率
        scheduler.step()
        # 打印当前学习率（可选）
        print(f"Epoch {epoch+1}, LR: {scheduler.get_last_lr()[0]}")
        train_losses.append(running_loss / len(trainloader))
        epoch_accuracy = 100 * correct / total  # 计算这个周期的准确率
        train_accuracies.append(epoch_accuracy)  # 将这个周期的准确率添加到列表中
        update_plots(train_losses, train_accuracies)	# 更新图表
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}, Accuracy: {epoch_accuracy:.2f}%")
    return train_losses
```

### 定义评估函数

```py
def test_model(model, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            # torch.max 函数用于返回指定维度上的最大值。
            # 在这里，我们希望得到每个验证码字符位置的最大分数对应的字符类别的索引。
            # 因为 outputs 的形状是 [batch_size, sequence_length, num_chars]，我们沿着第 2 个维度（索引为2）取最大值，这样我们可以为每个样本的每个字符位置得到一个最大分数的索引。
            _, predicted = torch.max(outputs.data, 2)
            
            # 因为 labels 的维度是 [batch_size, sequence_length]
            # 这一行代码计算了总的字符数量。labels.size(0) 是批次中的样本数，而 labels.size(1) 是每个验证码的字符长度。
            # 所以，两者的乘积给出了这个批次中所有验证码字符的总数。
            total += labels.size(0) * labels.size(1)
            
            # predicted == labels 返回一个布尔值的张量，形状与 labels 相同。如果预测正确，对应位置的值为 True，否则为 False。
            # .sum() 计算这个批次中正确预测的字符数量。
            # .item() 方法将单元素张量的值转换为 Python 数值。
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
```

### 定义绘制函数
定义绘制函数，用于绘制训练过程中的损失和准确率

```py
def plot_training(train_losses, test_accuracies):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(train_losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Test Accuracy', color=color)
    ax2.plot(test_accuracies, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()
```

### 训练函数的流程

```py
epochs = 10

# 训练模型
train_losses = train_model(
    model, trainloader, criterion, optimizer, epochs, device)

# 测试模型
test_accuracies = [test_model(model, testloader, device)
                   for _ in range(epochs)]

# 绘制训练损失和测试精度
plot_training(train_losses, test_accuracies)

print("Finished Training")

# 评估模型
test_accuracy = test_model(model, testloader, device)
print(f"Accuracy on test set: {test_accuracy}%")

# 保存模型
torch.save(model.state_dict(), 'captcha_model.pth')
```


## 补充：使用 TensorBoard
要使用 TensorBoard 与 PyTorch 配合，需要使用 `torch.utils.tensorboard` 包。

1. 导入 `SummaryWriter` 用于 TensorBoard 的日志。
2. 创建一个 `SummaryWriter` 对象。
3. 在训练循环中使用 `writer.add_scalar` 记录损失和准确率。
4. 在训练结束后，关闭 `writer`。

下面是修改后的代码部分。我只修改了与 TensorBoard 相关的部分，其他代码保持不变。

```python
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# ... [其它代码保持不变]

# 创建一个 SummaryWriter 对象，这里目录得不一样
now = datetime.now()
logdir = "runs/" + now.strftime("%Y%m%d-%H%M%S") + ""
writer = SummaryWriter(logdir)

# ... [其它代码保持不变]

def train_model(model, trainloader, criterion, optimizer, epochs, device):
    train_losses = []
    train_accuracies = []
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for _, data in enumerate(trainloader, 0):
            # ... [其它代码保持不变]

        scheduler.step()
        
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100 * correct / total

        # 使用 SummaryWriter 记录损失和准确率
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', epoch_accuracy, epoch)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # 打印日志，不再更新图表
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy:.2f}%")
        
    return train_losses

# ... [其它代码保持不变]

# 在所有训练和评估完成后关闭 writer
writer.close()
```

现在使用 TensorBoard 来查看损失和准确率的曲线图。

运行 TensorBoard:

```bash
tensorboard --logdir=runs
```

在浏览器中访问 TensorBoard (通常是 http://localhost:6006/) 来查看您的训练损失和准确率的曲线图。


## 定义模型
深度学习debug的流程策略

先从简单模型入手，然后逐渐增加模型的复杂度。把这个过程分为5个步骤：

1. 从最简单模型入手；
2. 成功搭建模型，重现结果；
3. 分解偏差各项，逐步拟合数据；
4. 用由粗到细随机搜索优化超参数；
5. 如果欠拟合，就增大模型；如果过拟合，就添加数据或调整。


### 第一版模型
首先定义一个最基本的 LeNet 模型

```py
class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        """验证码识别模型

        Args:
            num_chars (int): 字符类别的数量
        """
        super(CaptchaModel, self).__init__()

        # 假设验证码由4个字符组成
        self.num_chars = num_chars
        # 输入的是 3 维的图像，输出的是 32 维的特征图
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 对于 128x128 输入，经过4次下采样后，尺寸将变为 8x8
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        # 输出层，把 1024 维的特征转换为 4 * num_chars 维
        self.fc2 = nn.Linear(1024, self.num_chars*4)  # 4 是验证码包含的字符数

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(-1, 256 * 8 * 8)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x.view(x.size(0), 4, self.num_chars)
```

训练了 10 轮，发现曲线是下面这样的，可以发现 Loss 持续下降，但是 Accuracy 波动很大

![](https://image.alsritter.icu/img202310171043630.png)

### 调整数据集
上面取得的是数据集的 RGB 三个通道，但是我们的验证码的背景颜色和字体颜色都是随机的，所以我们可以转成灰度图，这样可以减少模型的复杂度，也可以加快训练速度。

```py
# 数据加载和预处理
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),  # 转换为RGB格式
    transforms.Grayscale(num_output_channels=1),    # 转换为灰度图像
    transforms.Lambda(pad_to_square),               # 将图像填充为正方形
    transforms.Resize((128, 128)),                  # 将图像调整为所需尺寸
    transforms.ToTensor(),
])
```

简单修改一下模型，把输入通道改成 1

```py
self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
```

可以看到下面的这个曲线平滑了不少（红色的线），但是 Loss 降的很慢，Accuracy 也不高

![](https://image.alsritter.icu/img202310171138998.png)

这个 Loss 下降到一定程度就不再下降了，这个现象叫做欠拟合（Underfitting），这是因为模型的复杂度不够，无法拟合数据。所以下面换一个更复杂的模型。

### 第二版模型
为了增加模型的复杂性以适应复杂的验证码，我们可以考虑以下策略：

1. 增加卷积层的深度：添加更多的卷积层可以帮助模型提取更复杂的特征。
2. 加入 Batch Normalization：它可以加速训练并提高模型的稳定性。
3. 加入 Dropout：防止过拟合，尤其在全连接层中。
4. 使用 Adaptive Pooling：这样模型可以适应不同大小的输入。

以下是根据这些策略修改后的模型代码：

```py
class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()

        self.num_chars = num_chars

        # 增加卷积层的深度
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)

        # 使用Adaptive Pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, self.num_chars*4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.adaptive_pool(F.relu(self.bn5(self.conv5(x))))

        x = x.view(-1, 512)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 在全连接层后增加 Dropout
        x = self.fc2(x)

        return x.view(x.size(0), 4, self.num_chars)
```

可以看到提升了模型复杂度后，Loss 下降的速度加快了，Accuracy 也提升了不少

![](https://image.alsritter.icu/img202310171156421.png)

看起来之前是因为模型太简单了，对这种复杂的分类任务无法拟合，所以才会出现欠拟合的现象。下面再尝试更复杂的模型。

### 第三版模型
这里直接尝试 ResNet 残差网络

```py
import torch.nn.functional as F
import torch.nn as nn

# 定义一个残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # 主要的卷积路径
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二层卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 快捷路径（shortcut）
        # 当输入输出的维度不一致时，我们需要调整其维度（可能是由于 stride 不为 1 或 in_channels != out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 1x1卷积用于调整维度
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 添加快捷路径
        out = F.relu(out)
        return out

class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()

        self.num_chars = num_chars
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 残差块
        self.res_block1 = ResidualBlock(32, 64)
        self.res_block2 = ResidualBlock(64, 128)
        self.res_block3 = ResidualBlock(128, 256)
        self.res_block4 = ResidualBlock(256, 512)
        
        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc1 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.5)  # 防止过拟合
        self.fc2 = nn.Linear(1024, self.num_chars*4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.res_block1(x))  # 通过残差块
        x = self.pool(self.res_block2(x))
        x = self.pool(self.res_block3(x))
        x = self.adaptive_pool(self.res_block4(x))
        
        x = x.view(-1, 512)  # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.view(x.size(0), 4, self.num_chars)  # 重新调整形状以匹配预期的输出形状
```

再次看曲线，发现这个曲线相较之前的模型，简直就是碾压般的存在，Loss 下降的很快，Accuracy 也很高

![](https://image.alsritter.icu/img202310171232765.png)

可以看到随着模型的复杂度提升，训练的时间也增长了

![](https://image.alsritter.icu/img202310171233243.png)

### 调整 Batch Size
上面设置 Batch Size 为 64，实际上越大的 Batch Size，训练的速度越快，但是模型的泛化能力会下降，所以我们可以尝试调整 Batch Size，看看会发生什么。

:::tip
实际上 batch size 可以说是所有超参数里最好调的一个，也是应该最早确定下来的超参数。一般是先选好 batch size，再调其他的超参数。上面是为了体现不同模型的效果，所以把它放在这里了。
:::

先调整成 256

```py
train_dataset = CaptchaDataset(root_dir='captcha', transform=transform)
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=256, shuffle=True)

test_dataset = CaptchaDataset(root_dir='test_captcha', transform=transform)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=256, shuffle=True)
```

首先可以发现 GPU 占用率从之前的 40% 飙升到 70% 了，这是因为 Batch Size 增大了，所以 GPU 的计算负载也增大了。

![](https://image.alsritter.icu/img202310171307113.png)

调整成 256 之后，发现训练时间是下来了（没有下降太多），但是准确性也同样的下降了，这是因为 Batch Size 增大了，模型的泛化能力下降了。

![](https://image.alsritter.icu/img202310171326472.png)

![](https://image.alsritter.icu/img202310171327388.png)

再改成 128 发现还是不行，所以改回 64

![](https://image.alsritter.icu/img202310171334755.png)

### 调整 Epochs
为了提高准确性，这里就开始提高轮次，从 10 轮提高到 50 轮

![](https://image.alsritter.icu/img202310171527468.png)

可以发现从 30 轮开始，Loss 下降的速度就很慢了，这是因为模型已经收敛了，再训练就没有意义了，所以这里就停止训练了。

![](https://image.alsritter.icu/img202310171528391.png)

最终训练的结果是 50% 正确率，有点低，下面尝试再提升训练集的大小。使用 2w 张图片训练，1w 张图片测试，因为数据量比较大，所以批次大小改成 128

![](https://image.alsritter.icu/img202310172051913.png)

发现这两条曲线的趋势是一致的，说明还是因为模型不够复杂，到上面 30 轮就到极限了，以至于无法拟合数据，所以这里就换一个更复杂的模型。

### 第四版模型



## References
* [为什么你的模型效果这么差，深度学习调参有哪些技巧？](https://zhuanlan.zhihu.com/p/165417543)
* [深度学习欠拟合问题的判断和解决](https://blog.csdn.net/qq_34044577/article/details/105705344)