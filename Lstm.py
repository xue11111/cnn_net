import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.arange(10).reshape((2,5)).float().to(device)

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device, requires_grad=True) * 0.01

    def three():
        return (normal((num_inputs,num_hiddens)),
                normal((num_hiddens,num_hiddens)),
                torch.zeros(num_hiddens,device=device,requires_grad=True))

    w1, w2, b1 = three()
    w3, w4, b2 = three()
    w5, w6, b3 = three()
    w7, w8, b4 = three()

    w_out = normal((num_hiddens, num_outputs))
    b_out = torch.zeros(num_outputs,  device=device,requires_grad=True)

    return [w1, w2, b1, w3, w4, b2, w5, w6, b3,w7, w8, b4, w_out, b_out]

def init_lstm_state(batch_size,num_hiddens, device):
    return (torch.zeros((batch_size,num_hiddens), device=device),
            torch.zeros((batch_size,num_hiddens), device=device))

def lstm(inputs,state,params):
    w1, w2, b1, w3, w4, b2, w5, w6, b3,w7, w8, b4, w_out, b_out = params
    H, C = state
    outputs = []
    for x in inputs:
        #矩阵乘法
        I = torch.sigmoid((x @ w1) + (H @ w2) + b1)
        F = torch.sigmoid((x @ w3) + (H @ w4) + b2)
        O = torch.sigmoid((x @ w5) + (H @ w6) + b3)
        C_ = torch.tanh((x @ w7) + (H @ w8) + b4)
        C = F * C + I * C_
        H = O * torch.tanh(C)
        Y = (H @ w_out) + b_out
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)

params = get_params(5,256,device)
state = init_lstm_state(2,256,device)
outputs,state = lstm(x,state,params)
print(outputs.shape)