import torch
import torch.nn as nn
from d2l import torch as d2l


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trian_iter,vocab = d2l.load_data_time_machine(
    32,35)

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape,device=device,requires_grad=True) * 0.01

    def three():
        return (normal((num_inputs,num_hiddens)),
                normal((num_hiddens,num_hiddens)),
                torch.zeros(num_hiddens,device=device,requires_grad=True))

    w1, w2, b1 = three()
    w3, w4, b2 = three()
    w5, w6, b3 = three()

    w_out = normal((num_hiddens,num_outputs))
    b_out = torch.zeros(num_outputs,device=device,requires_grad=True)

    return [w1, w2, b1, w3, w4, b2, w5, w6, b3, w5, w6, b3, w_out, b_out]

def init_gru_state(batch_size,num_hiddens, device):
    return (torch.zeros((batch_size,num_hiddens), device=device), )

def gru(inputs,state,params):
    w1, w2, b1, w3, w4, b2, w5, w6, b3, w5, w6, b3, w_out, b_out = params
    H, = state
    outputs = []
    for x in inputs:
        #矩阵乘法
        z = torch.sigmoid((x @ w1) + (H @ w2) + b1)
        r = torch.sigmoid((x @ w3) + (H @ w4) + b2)
        H_tilda = torch.tanh((x @ w5) + ((r * H) @ w6) + b3)
        #矩阵内积
        H = z * H +(1 - z) * H_tilda
        Y = H @ w_out + b_out
        outputs.append(Y)
    return torch.cat(outputs,dim=0), (H,)

