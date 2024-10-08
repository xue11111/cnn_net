import torch
import numpy as np
import torch.optim as optimizer
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt


sentences = ["jack like dog", "jack like cat", "jack like animal",
  "dog cat animal", "banana apple cat dog like", "dog fish milk like",
  "dog cat animal like", "jack like apple", "apple like", "jack like banana",
  "apple banana jack movie book music like", "cat dog hate", "cat dog like"]

# ['apple', 'banana', 'fruit', 'banana', 'orange', 'fruit', 'orange', 'banana', 'fruit', 'dog', 'cat', 'animal', 'cat', 'monkey', 'animal', 'monkey', 'dog', 'animal']
word_sentence = " ".join(sentences).split()
# ['orange', 'cat', 'monkey', 'banana', 'fruit', 'animal', 'dog', 'apple']
vocab = list(set(word_sentence))
word2idx = {token: idx for idx, token in enumerate(vocab)}

dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 窗口大小
c = 2
batch_size = 8
h = 2

skip_gram = []
center = []
bg_context = []
for idx in range(c, len(word_sentence) - c):
    token = word_sentence[idx]
    # 中心词
    center += [word2idx[token]]
    tmp = []
    for i in range(idx - c, idx + c + 1):
        if i != idx:
            bg_token = word_sentence[i]
            tmp.append(word2idx[bg_token])
    # 中心词
    bg_context.append(tmp)

for idx in center:
    for list in bg_context:
        for i in list:
            skip_gram.append([idx, i])

input_data = []
output_data = []

for list in skip_gram:
    input_data.append(np.eye(len(vocab))[list[0]])
    output_data.append(list[1])

# 将输入数据转换成浮点型tensor 输出数据转换成长整型tensor，常用于分类标签
input_data, output_data = torch.Tensor(input_data),torch.LongTensor(output_data)
dataset = Data.TensorDataset(input_data,output_data)
dataloader = Data.DataLoader(dataset,batch_size,True)

class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()

        self.W = nn.Parameter(torch.randn(len(vocab), h)).type(dtype)
        self.V = nn.Parameter(torch.randn(h, len(vocab))).type(dtype)


    def forward(self, x):
        hidden = torch.mm(x, self.W)
        output = torch.mm(hidden, self.V)
        return output

model = Word2Vec().to(device)

loss = nn.CrossEntropyLoss().to(device)
optim = optimizer.Adam(model.parameters(), lr=1e-3)

for epoch in range(200):
    l = 0
    for input_data, output_data in dataloader:
        input_data = input_data.to(device)
        output_data = output_data.to(device)
        pred = model(input_data)
        loss_fn = loss(pred, output_data)
        optim.zero_grad()
        loss_fn.backward()
        optim.step()
        l += loss_fn

    print("第{}轮的损失：".format(epoch), l)

for i, label in enumerate(vocab):
  W, WT = model.parameters()
  x,y = float(W[i][0]), float(W[i][1])
  plt.scatter(x, y)
  plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()