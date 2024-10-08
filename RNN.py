
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from d2l import torch as d2l


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trian_iter,vocab = d2l.load_data_time_machine(
    32,35)

x = torch.arange(10).reshape((2,5))

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape,device=device) * 0.01

    w_xh = normal((num_inputs,num_hiddens))
    w_hh = normal((num_hiddens,num_hiddens))
    b_xh = torch.zeros(num_hiddens,device=device)

    w_hq = normal((num_hiddens,num_outputs))
    b_hq = torch.zeros(num_outputs,device=device)
    params = [w_xh,w_hh,b_xh,w_hq,b_hq]

    for param in params:
        param.requires_grad_(True)
    return params


#初始化隐藏变量
def init_run_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device=device),)



def run(inputs,state,params):
    w_xh,w_hh,b_xh,w_hq,b_hq = params
    H, = state
    output = []

    for x in inputs:
        H = torch.tanh(torch.mm(x,w_xh) + b_xh + torch.mm(H,w_hh))
        y = torch.mm(H,w_hq) + b_hq
        output.append(y)
    return torch.cat(output,dim=0),(H,)

class RNNModelScratch:
    def __init__(self,vocab_size,num_hiddens,device,get_params,init_state,forward_fn):
        self.vocab_size,self.num_hiddens = vocab_size,num_hiddens
        self.params = get_params(vocab_size,num_hiddens,device)
        self.init_state,self.forward_fn = init_state,forward_fn

    def __call__(self,x,state):
        x = F.one_hot(x.T,self.vocab_size).type(torch.float32)
        return self.forward_fn(x,state,self.params)

    def begin_state(self,batch_size,device):
        return self.init_state(batch_size,self.num_hiddens,device)

num_hiddens = 512
net = RNNModelScratch(len(vocab),num_hiddens,device,get_params,init_run_state,run)

# state = net.begin_state(x.shape[0],device)
# y,new_state = net(x.to(device),state)
# print(y.shape,len(new_state),new_state[0].shape)
def predict_ch8(prefix,num_preds,net,vocab,devicec):
    state = net.begin_state(batch_size=1,device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda : torch.tensor([outputs[-1]],device=device).reshape((1,1))

    for y in prefix[1:]:
        _,state = net(get_input(),state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y,state = net(get_input(),state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# pred = predict_ch8('time traveller ',10,net,vocab,device)

def grad_clipping(net, theta):
    if isinstance(net,nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            for s in state:
                s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y).mean()
        if isinstance(updater,torch.optim.Optimizer):
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net,1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device,  #@save
              use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(),lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller '))
    print(predict('traveller '))

num_epochs, lr = 500, 1
train_ch8(net, trian_iter, vocab, lr, num_epochs, device)



