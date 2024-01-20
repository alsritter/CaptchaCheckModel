This experiment aims to explore the effectiveness of Convolutional Neural Networks (CNN) in captcha recognition. Considering captcha complexity, CNN is used to develop a model that recognizes various captchas (including distorted text, noisy backgrounds, etc.) and assess its practical application feasibility.

## Generating the Dataset
For a preliminary trial, we opt for a simpler captcha generating library:

```bash
go get github.com/afocus/captcha
```

Checking the installed fonts:

```bash
# For Linux, use fc-list
dir C:\Windows\Fonts
```

Writing a function to generate the dataset:

```go
# Linux 可以使用 fc-list
dir C:\Windows\Fonts
```


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

Generated captchas:

![](https://image.alsritter.icu/img202310131636897.png)

## Introducing Dependencies
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

## Data Preprocessing
Since the captchas aren't square, to preprocess the images into square shapes, we first perform padding and then scaling. Padding ensures the images have equal width and height without altering the content.

To achieve a 128x128 square image:
1. Use `transforms.Resize` to scale the shorter side of the image, maintaining its aspect ratio.
2. Employ `transforms.Pad` to make the longer side equal to the shorter one, achieving a square shape.
3. Finally, apply `transforms.Resize` again to adjust to the desired size.

Example code:

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

This method first pads the image into a square shape, then resizes it. This approach ensures the content of the image isn't distorted due to varying scaling ratios, maintaining consistency in the image content.

## Defining the Dataset Class
As the dataset is self-generated, we need to write our dataset collection class using PyTorch's `torch.utils.data.Dataset`, implementing the `__len__` and `__getitem__` methods.

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

## Defining Related Functions
Next, we define functions for training, evaluation, and plotting.

### Defining the Loss Function and Optimizer
First, we define the loss function and optimizer.

```py
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
# 定义StepLR调度器，每10个epoch，学习率乘以gamma=0.9
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
```

### Plotting Convergence per Epoch
Defining a function to plot convergence for each epoch.

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

### Defining the Training Function
Then, we define the training function, which is a standard training process.

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

### Defining the Evaluation Function

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

### Defining the Plotting Function
Defining a function to plot training loss and accuracy during the training process.

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

### Training Process Flow

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

## Supplement: Using TensorBoard
To integrate TensorBoard with PyTorch, use the `torch.utils.tensorboard` package.

1. Import `SummaryWriter` for logging in TensorBoard.
2. Create a `SummaryWriter` object.
3. Record loss and accuracy in the training loop using `writer.add_scalar`.
4. Close the `writer` after training.

Here's the modified code section, with changes only related to TensorBoard.

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

Now, use TensorBoard to view the graphs for training loss and accuracy.

To run TensorBoard:

```bash
tensorboard --logdir=runs
```

Access TensorBoard in the browser (usually at http://localhost:6006/) to view the graphs for training loss and accuracy.

## Defining the Model
Strategies for debugging in deep learning:

Start with a simple model and gradually increase its complexity. Divide this process into five steps:

1. Start with the simplest model.
2. Successfully build the model and reproduce results.
3. Decompose bias terms and progressively fit the data.
4. Use coarse-to-fine random search for hyperparameter optimization.
5. Increase model size for underfitting or add data/adjust for overfitting.

### First Version of the Model
First, we define a basic LeNet model.

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

After training for 10 epochs, we find the curve as shown below. The Loss continues to decrease, but the Accuracy fluctuates greatly.

![](https://image.alsritter.icu/img202310171043630.png)

### Adjusting the Dataset
Initially, we used three channels of the dataset, but since both the background and font colors of our captchas

 are random, we can convert them to grayscale. This reduces the model's complexity and speeds up training.

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

Simply modify the model to change the input channel to 1.

```py
self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
```

As seen in the smoother curve (red line) below, the Loss decreases slowly, and the Accuracy is not high.

![](https://image.alsritter.icu/img202310171138998.png)

The Loss stops decreasing beyond a point, indicating underfitting. This suggests the model is too simple for this complex classification task. Therefore, we switch to a more complex model.

### Second Version of the Model
To accommodate the complex captcha, we consider:

1. Increasing convolutional depth.
2. Adding Batch Normalization and Dropout.
3. Using Adaptive Pooling.

Here's the modified model code:

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

The Loss drops faster, and Accuracy improves significantly.

![](https://image.alsritter.icu/img202310171156421.png)

However, as model complexity increases, so does the training time.

![](https://image.alsritter.icu/img202310171233243.png)

### Adjusting Batch Size
Initially set to 64, we experiment with larger Batch Sizes.

:::tip
In fact, batch size is the easiest of all hyperparameters to adjust and should be determined first. It's usually chosen first before adjusting other hyperparameters.:::

First adjusting to 256:

```py
train_dataset = CaptchaDataset(root_dir='captcha', transform=transform)
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=256, shuffle=True)

test_dataset = CaptchaDataset(root_dir='test_captcha', transform=transform)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=256, shuffle=True)
```

GPU usage increases significantly, indicating a higher computational load due to the larger Batch Size.

![](https://image.alsritter.icu/img202310171307113.png)

After adjusting to 256, we notice a decrease in training time but also a drop in accuracy. This is due to the larger Batch Size reducing the model's generalization ability.

![](https://image.alsritter.icu/img202310171326472.png)

Adjusting to 128 doesn't work either, so we revert to 64.

![](https://image.alsritter.icu/img202310171334755.png)

### Adjusting Epochs
To improve accuracy, we increase the epochs from 10 to 50.

![](https://image.alsritter.icu/img202310171527468.png)

Beyond 30 epochs, the Loss decreases slowly, indicating model convergence. Thus, training is stopped.

![](https://image.alsritter.icu/img202310171528391.png)

The final result is a 50% accuracy rate, which is low. Therefore, we attempt to increase the training dataset size.

![](https://image.alsritter.icu/img202310172051913.png)

As model complexity increases, the Loss decreases rapidly, and Accuracy improves significantly.

![](https://image.alsritter.icu/img202310171232765.png)

### Fourth Version of the Model
We now try a ResNet residual network for even greater complexity.

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

After training with the new model, the curve shows significant improvement over previous models.

![](https://image.alsritter.icu/img202310171232765.png)

The trade-off is increased training time due to the model's complexity.

![](https://image.alsritter.icu/img202310171233243.png)

## References
* [Why is your model performing poorly, and what are the techniques for tuning deep learning?](https://zhuanlan.zhihu.com/p/165417543)
* [Solving and Identifying Underfitting in Deep Learning](https://blog.csdn.net/qq_34044577/article/details/105705344)