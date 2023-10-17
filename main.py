import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    print("CUDA (GPU support) is available and PyTorch can use GPUs!")
else:
    print("CUDA is not available. PyTorch will use CPU.")

# 检查 CUDA 是否可用并定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


# 数据加载和预处理
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),  # 转换为RGB格式
    transforms.Grayscale(num_output_channels=1),    # 转换为灰度图像
    transforms.Lambda(pad_to_square),               # 将图像填充为正方形
    transforms.Resize((128, 128)),                  # 将图像调整为所需尺寸
    transforms.ToTensor(),
])

# 定义分类
nums = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


# 创建字符到整数的映射
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


train_dataset = CaptchaDataset(root_dir='captcha', transform=transform)
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True)

test_dataset = CaptchaDataset(root_dir='test_captcha', transform=transform)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=True)



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()

        self.num_chars = num_chars
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.res_block1 = ResidualBlock(32, 64)
        self.res_block2 = ResidualBlock(64, 128)
        self.res_block3 = ResidualBlock(128, 256)
        self.res_block4 = ResidualBlock(256, 512)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, self.num_chars*4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.res_block1(x))
        x = self.pool(self.res_block2(x))
        x = self.pool(self.res_block3(x))
        x = self.adaptive_pool(self.res_block4(x))
        
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.view(x.size(0), 4, self.num_chars)

model = CaptchaModel(len(nums)).to(device)

# 创建一个 SummaryWriter 对象
now = datetime.now()
logdir = "runs/" + now.strftime("%Y%m%d-%H%M%S") + ""
writer = SummaryWriter(logdir)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# 定义StepLR调度器，每10个epoch，学习率乘以gamma=0.9
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)


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
        # 打印当前学习率
        print(f"Epoch {epoch+1}, LR: {scheduler.get_last_lr()[0]}")

        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100 * correct / total  # 计算这个周期的准确率

        # 使用 SummaryWriter 记录损失和准确率
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', epoch_accuracy, epoch)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)  # 将这个周期的准确率添加到列表中

        # 打印日志
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy:.2f}%")
    
    # 在 TensorBoard 中记录模型的结构
    inputs, _ = next(iter(trainloader))
    inputs = inputs.to(device)
    writer.add_graph(model, inputs)

    return train_losses


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



epochs = 10

# 训练模型
train_losses = train_model(
    model, trainloader, criterion, optimizer, epochs, device)


print("Finished Training")

# 评估模型
test_accuracy = test_model(model, testloader, device)
print(f"Accuracy on test set: {test_accuracy}%")


writer.close()

# 保存模型
torch.save(model.state_dict(), 'captcha_model.pth')