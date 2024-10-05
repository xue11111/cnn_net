import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torchvision import transforms



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.module = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.AvgPool2d(2,2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10)
        )

    def forward(self, x):
        return self.module(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)

# 初始化权重

for layer in model.module:
    if isinstance(layer,(nn.Conv2d,nn.Linear)):
        nn.init.xavier_uniform_(layer.weight)


PIL_totensor = transforms.Compose(
    [transforms.Resize((32,32)),
     transforms.ToTensor()]
)

train_dataset = torchvision.datasets.MNIST(root="./data",train=True,transform=PIL_totensor,download=True)
test_data = torchvision.datasets.MNIST(root="./data",train=False,transform=PIL_totensor,download=True)


train_loader = Data.DataLoader(train_dataset,batch_size=8)
test_loader = Data.DataLoader(test_data,batch_size=8)


loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



# 初始化损失列表
train_losses = []
test_losses = []

for epoch in range(5):
    total_train_loss = 0
    total_test_loss = 0

    model.train()
    for img, target in train_loader:
        img = img.to(device)
        target = target.to(device)
        loss_train = loss_fn(model(img), target)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        total_train_loss += loss_train.item()  # 使用 .item() 获取标量值

    train_losses.append(total_train_loss / len(train_loader))  # 平均损失
    print("第{}轮训练损失为：".format(epoch),total_train_loss / len(train_loader))

    model.eval()
    with torch.no_grad():
        for img, target in test_loader:
            img = img.to(device)
            target = target.to(device)
            loss_test = loss_fn(model(img), target)
            total_test_loss += loss_test.item()  # 使用 .item() 获取标量值

    test_losses.append(total_test_loss / len(test_loader))  # 平均损失
    print("第{}轮验证损失为：".format(epoch), total_test_loss / len(test_loader))

plt.rcParams['font.family'] = 'SimHei'  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 处理负号显示问题
# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='训练损失')
plt.plot(test_losses, label='验证损失')
plt.title('训练与验证损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()
plt.grid()
plt.show()





