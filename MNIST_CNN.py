import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

lossvalues = []

# 先定義好下載了的MNIST data 要有什麼處理。
# 這兒有兩個動作 (1)convert成PyTorch的tensor
# (2)基於mean=0.5, std=0.5 進行normalization, 注意!因為是單色相片，只有一
# 個值，如果是RGB相，會是 (0.5,0.5,0.5)
# ----------------------------------------
#print("GPU:", torch.cuda.is_available())
# _tasks = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5,), std=(0.5,))
# ])
_tasks = transforms.Compose([
    transforms.ToTensor(),
])
# 告訴它，你要下載到'data'檔案夾中。
# ----------------------------------------
mnist = MNIST("data", download=True, train=True, transform=_tasks)
split = int(0.8 * len(mnist))
index_list = list(range(len(mnist)))

# print(mnist)
train_idx, valid_idx = index_list[:split], index_list[split:]
# 用 SubsetRandomSampler，可向它拿取一個 random 的 element.
# ----------------------------------------
tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)
# train_idx 有 48000 個 element
# valid_idx 有 12000 個 element
# 以下batch_size=256,即48000 element 中每次抽256個 samples 出來train。
#     它們產生的error, 平均了之後才去update weight。
# 以下batch_size=256,即12000 element 中每次抽256個 samples 出來train。
#     它們產生的error, 平均了之後才去update weight。
# ----------------------------------------
trainloader = DataLoader(mnist, batch_size=256, sampler=tr_sampler)
validloader = DataLoader(mnist, batch_size=256, sampler=val_sampler)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
print(device)

for batch_idx, (img, label) in enumerate(trainloader):
    img = img.to(device)
    label = label.to(device)


class Model(nn.Module):  # 必要 inherit nn.Module
    def __init__(self):
        super().__init__()
        # 輸入通道 = 1，輸出通道 = 4，kernal size = 5，padding補齊feature map
        self.cnn1 = nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)  # 2*2 maxpool
        self.cnn2 = nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2)
        self.output1 = nn.Linear(28//4 * 28//4 * 8, 512)
        self.output2 = nn.Linear(512, 10)
        # self.hidden = nn.Linear(784, 128)
        # # self.output2 = nn.Linear(128, 128)
        # # self.output3 = nn.Linear(128, 87)
        # self.output = nn.Linear(128, 10)
    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(-1, 28//4 * 28//4 * 8)  # flattening
        x = torch.relu(self.output1(x))  # fully connected
        # x = torch.dropout(x, 0.5)  # 機率0.5dropout
        x = self.output2(x)  # x = (batch size * 10)
        x = torch.log_softmax(x, dim=1)
        # x = self.hidden(x)
        # # x = self.output2(x)
        # # x = self.output3(x)
        # x = torch.relu(x)  # 當然可以改用其他activation。
        # x = self.output(x)
        return x
    # def retrieve_features(self, x):


model = Model()
loss_function = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01,
#   weight_decay=1e-6, momentum=0.9, nesterov=True)

optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 10):
    train_loss, valid_loss = [], []
    ## 把 model 調去 training mode;##
    model.train()
    ## trainloader 中的每個batch 有256個 samples. ##
    for data, target in trainloader:

        optimizer.zero_grad()  # optimizer 中的gradient 先歸零.#
        # 1. forward propagation
        # -----------------------

        # 把本來 torch.Size([256, 1, 28, 28])
        # 用 .view transform 成 torch.Size([256, 784])
        # data = data.view(data.size(0), -1)
        # 這個class 是callable 的。而每次call 都會運行 forward().
        # 自Appendix 1.
        output = model(data)
        # 2. loss calculation
        # -----------------------
        loss = loss_function(output, target)
        # optimizer.zero_grad
        # 3. backward propagation
        # -----------------------
        # 是不是很神，為什麼loss.backward() 就backward propagation 了，
        # 它知道要backward 給誰嗎? 是的，它就是知道。請看 blog [ML13]
        loss.backward()

        # 4. weight optimization
        # -----------------------
        optimizer.step()
        train_loss.append(loss.item())

    # evaluation part
    # 把 model 調去 training mode;
    model.eval() 
    for data, target in validloader:
        # data = data.view(data.size(0), -1)
        output = model(data)
        loss = loss_function(output, target)
        valid_loss.append(loss.item())

        print("Epoch:", epoch, "Training Loss: ",  np.mean(
            train_loss), "Valid Loss: ", np.mean(valid_loss))
        lossvalues.append(np.mean(valid_loss))

# dataloader for validation dataset
dataiter = iter(validloader)
data, labels = dataiter.next()
print(data.shape)
# data = data.view(data.size(0), -1)
output = model(data)
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy())
np.save('0723_1.npy', lossvalues)
print("Actual:", labels[:10])
print("Predicted:", preds[:10])
torch.save(model, './cnn.pt')
