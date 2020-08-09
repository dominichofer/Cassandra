import struct
import numpy
import time
import math
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import wandb
from torch.utils.data import DataLoader, TensorDataset
from Position import *
from ConvNN import *

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def DataGenerator(empties:int, num:int):
    chunk_size = struct.calcsize('<QQb')
    with open("G:\\Reversi\\rnd\\e{}.psc".format(empties), "rb") as file:
        for _ in range(num):
            try:
                P, O, score = struct.unpack('<QQb', file.read(chunk_size))
                yield Position(P, O), score
            except:
                break
    
def GetData(empties:list, test_size:int, train_size:int):
    test = []
    train = []
    for e in empties:
        data = list(DataGenerator(e, test_size + train_size))
        test.extend(data[:test_size])
        train.extend(data[test_size:])
    return test, train


def to_bit_list(pos:Position):
    return [1 if pos.P[i, j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.O[i, j] else 0 for i in range(8) for j in range(8)]

def to_ternary_list(pos:Position):
    return [1 if pos.P[i, j] else (-1 if pos.O[i, j] else 0) for i in range(8) for j in range(8)]

def to_bit_list_with_symmetrie(pos:Position):
    return [1 if pos.P[i, j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.O[i, j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.P[i, 7-j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.O[i, 7-j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.P[7-i, j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.O[7-i, j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.P[7-i, 7-j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.O[7-i, 7-j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.P[i, j] else 0 for j in range(8) for i in range(8)] \
         + [1 if pos.O[i, j] else 0 for j in range(8) for i in range(8)] \
         + [1 if pos.P[i, 7-j] else 0 for j in range(8) for i in range(8)] \
         + [1 if pos.O[i, 7-j] else 0 for j in range(8) for i in range(8)] \
         + [1 if pos.P[7-i, j] else 0 for j in range(8) for i in range(8)] \
         + [1 if pos.O[7-i, j] else 0 for j in range(8) for i in range(8)] \
         + [1 if pos.P[7-i, 7-j] else 0 for j in range(8) for i in range(8)] \
         + [1 if pos.O[7-i, 7-j] else 0 for j in range(8) for i in range(8)]

def to_tensor_2x8x8(data):
    pos, score = zip(*data)
    pos = torch.tensor([to_bit_list(x) for x in pos], dtype=torch.float).view(-1, 2, 8, 8)
    score = torch.tensor(score, dtype=torch.float)
    return pos, score

def to_tensor_2x8x8_with_symmetrie(data):
    pos, score = zip(*data)
    pos = torch.tensor([to_bit_list_with_symmetrie(x) for x in pos], dtype=torch.float).view(-1, 2, 8, 8)
    score = torch.tensor([x for x in score for _ in range(8)], dtype=torch.float)
    return pos, score

def to_tensor_binary(data):
    pos, score = zip(*data)
    pos = torch.tensor([to_bit_list(x) for x in pos], dtype=torch.float).view(-1, 128)
    score = torch.tensor(score, dtype=torch.float)
    return pos, score

def to_tensor_binary_with_symmetrie(data):
    pos, score = zip(*data)
    pos = torch.tensor([to_bit_list_with_symmetrie(x) for x in pos], dtype=torch.float).view(-1, 128)
    score = torch.tensor([x for x in score for _ in range(8)], dtype=torch.float)
    return pos, score

def to_tensor_ternary(data):
    pos, score = zip(*data)
    pos = torch.tensor([to_ternary_list(x) for x in pos], dtype=torch.float).view(-1, 64)
    score = torch.tensor(score, dtype=torch.float)
    return pos, score


def count_parameters(model):
    sum = 0
    for _, param in model.named_parameters():
        if param.requires_grad:
            sum += numpy.prod(param.size())
    return sum


class Plotter:
    def __init__(self):
        self.epoch = []
        self.train = []
        self.test = []

        plt.ion()
        self.fig = plt.figure()
        self.sub = self.fig.add_subplot(111)
        self.plot()

    def plot(self):
        self.sub.plot(self.epoch, self.train, 'b-', self.epoch, self.test, 'r-')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def add(self, train, test):
        self.epoch.append(len(self.epoch))
        self.train.append(train)
        self.test.append(test)
        self.plot()

class ValueHead(nn.Module):
    def __init__(self, input, internal):
        super(ValueHead, self).__init__()
        self.layer1 = Conv1x1_BN_ReLU(input, 1)
        self.full = nn.Linear(internal, internal)
        self.shrink = nn.Linear(internal, 1)

    def forward(self, x):
        y = self.layer1(x).flatten(1, -1)
        y = F.relu(self.full(y), inplace=True)
        return self.shrink(y)

class SE_Layer(nn.Module):
    """ Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 """
    def __init__(self, planes, reduction):
        super(SE_Layer, self).__init__()
        hidden_planes = planes // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.full1 = nn.Linear(planes, hidden_planes, bias=False)
        self.full2 = nn.Linear(hidden_planes, planes, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = F.relu(self.full1(y), inplace=True)
        y = torch.sigmoid(self.full2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNet_Basic(nn.Module):
    """ https://arxiv.org/abs/1512.03385 """
    def __init__(self, in_planes, hidden_planes=None, out_planes=None):
        super(ResNet_Basic, self).__init__()
        if hidden_planes is None:
            hidden_planes = in_planes
        if out_planes is None:
            out_planes = in_planes
        self.layer1 = Conv3x3_BN_ReLU(in_planes, hidden_planes)
        self.layer2 = Conv3x3_BN(hidden_planes, out_planes)
        self.skip = None if in_planes == out_planes else Conv1x1_BN(in_planes, out_planes)

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        if self.skip is not None:
            x = self.skip(x)
        return F.relu(x + y, inplace=True)

class ResNet_Bottleneck(nn.Module):
    def __init__(self, in_planes, hidden_planes=None, out_planes=None):
        super(ResNet_Bottleneck, self).__init__()
        if hidden_planes is None:
            hidden_planes = in_planes
        if out_planes is None:
            out_planes = in_planes
        self.layer1 = Conv1x1_BN_ReLU(in_planes, hidden_planes)
        self.layer2 = Conv3x3_BN_ReLU(hidden_planes, hidden_planes)
        self.layer3 = Conv1x1_BN(hidden_planes, out_planes)
        self.skip = None if in_planes == out_planes else Conv1x1_BN(in_planes, out_planes)

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        if self.skip is not None:
            x = self.skip(x)
        return F.relu(x + y, inplace=True)

class SE_Basic(nn.Module):
    def __init__(self, in_planes, hidden_planes=None, out_planes=None, reduction=4):
        super(SE_Basic, self).__init__()
        if hidden_planes is None:
            hidden_planes = in_planes
        if out_planes is None:
            out_planes = in_planes
        self.layer1 = Conv3x3_BN_ReLU(in_planes, hidden_planes)
        self.layer2 = Conv3x3_BN(hidden_planes, out_planes)
        self.se = SE_Layer(out_planes, reduction)
        self.skip = None if in_planes == out_planes else Conv1x1_BN(in_planes, out_planes)

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.se(y)
        if self.skip is not None:
            x = self.skip(x)
        return F.relu(x + y, inplace=True)

class SE_Bottleneck(nn.Module):
    def __init__(self, in_planes, hidden_planes=None, out_planes=None, reduction=4):
        super(SE_Bottleneck, self).__init__()
        if hidden_planes is None:
            hidden_planes = in_planes
        if out_planes is None:
            out_planes = in_planes
        self.layer1 = Conv1x1_BN_ReLU(in_planes, hidden_planes)
        self.layer2 = Conv3x3_BN_ReLU(hidden_planes, hidden_planes)
        self.layer3 = Conv1x1_BN(hidden_planes, out_planes)
        self.se = SE_Layer(out_planes, reduction)
        self.skip = None if in_planes == out_planes else Conv1x1_BN(in_planes, out_planes)

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.se(y)
        if self.skip is not None:
            x = self.skip(x)
        return F.relu(x + y, inplace=True)
    
def ResNet(sizes:list, base_block):
    return nn.Sequential(
        Conv3x3_BN_ReLU(2, sizes[0]),
        *[base_block(i, o) for i, o in pairwise(sizes)],
        ValueHead(sizes[-1], 64)
        )

def Train(epochs, scheduler_step_size, lr, weight_decay, model, test, train, model_type, width, blocks):
    train_size = train[0].size()[0]
    test_size = test[0].size()[0]

    wandb.init(project="reversi", reinit=True)
    wandb.config.algorithm = "adam"
    wandb.config.epochs = epochs
    wandb.config.scheduler_step_size = scheduler_step_size
    wandb.config.learning_rate = lr
    wandb.config.weight_decay = weight_decay
    wandb.config.model_type = model_type
    wandb.config.model_width = width
    wandb.config.model_blocks = blocks
    wandb.config.model_parameters = count_parameters(model)
    wandb.config.train_size = train_size
    wandb.config.test_size = test_size
    wandb.watch(model)

    print('number of trainable parameters =', count_parameters(model))
    model = nn.DataParallel(model).to(device)

    test_loader = DataLoader(TensorDataset(*test), num_workers=8, batch_size=8*1024, shuffle=True, pin_memory=True)
    train_loader = DataLoader(TensorDataset(*train), num_workers=8, batch_size=8*1024, shuffle=True, pin_memory=True)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)

    for epoch in range(1,1+epochs):
        model.train()
        start_train = time.time()
        train_loss = 0
        for x, y in train_loader:
            x_size = x.size()[0]
            x = x.to(torch.device("cuda"))
            y = y.to(torch.device("cuda"))
            loss = loss_function(model(x).view(-1), y)
            train_loss += loss.item() * x_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = math.sqrt(train_loss / train_size)
        wandb.log({"train loss": train_loss})
        stop_train = time.time()
            
        with torch.no_grad():
            model.eval()
            start_test = time.time()
            test_loss = 0
            for x, y in test_loader:
                x_size = x.size()[0]
                x = x.to(torch.device("cuda"))
                y = y.to(torch.device("cuda"))
                loss = loss_function(model(x).view(-1), y)
                test_loss += loss.item() * x_size
            test_loss = math.sqrt(test_loss / test_size)
            wandb.log({"test loss": test_loss})
            stop_test = time.time()
        
        scheduler.step()
        print(epoch, train_loss, test_loss, stop_train - start_train, stop_test - start_test)
    wandb.join()
    return count_parameters(model), train_loss, test_loss
    

if __name__ == '__main__':
    device = torch.device("cuda")
    
    test_size = 50_000
    train_size = 950_000

    test, train = GetData([20], test_size, train_size)
    test = to_tensor_2x8x8(test)
    train = to_tensor_2x8x8(train)
    print('DataLoaded')
    
    results = []
    
    #for n in range(2, 7):
    #    for width in [16,32,64,128,256]:
    #        for lr in [0.01, 0.02, 0.05, 0.1]:
    #            results.append(Train(25, 20, lr, 0, ResNet([width]*n, ResNet_Basic), test, train, "ResNet_Basic", width, n-1))
    
    #for width in [64]:
    #    for n in range(14, 15):
    #        for lr in [0.01, 0.02, 0.1]:
    #            results.append(Train(25, 20, lr, 0, ResNet([width]*(n+1), ResNet_Bottleneck), test, train, "ResNet_Bottleneck", width, n))
          
    #for width in [64]:
    #    for n in range(11, 21):
    #        results.append(Train(25, 20, 0.05, 0, ResNet([width]*(n+1), SE_Bottleneck), test, train, "SE_Bottleneck", width, n))
            
    #for width in [128]:
    #    for n in range(1, 11):
    #        results.append(Train(25, 20, 0.05, 0, ResNet([width]*(n+1), SE_Bottleneck), test, train, "SE_Bottleneck", width, n))
            
    for width in [256]:
        for n in range(2, 6):
            results.append(Train(25, 20, 0.05, 0, ResNet([width]*(n+1), SE_Bottleneck), test, train, "SE_Bottleneck", width, n))


    #for n in range(2, 22):
    #    results.append(Train(55, 50, 0.02, 0, ResNet([64]*n, ResNet_Basic), test, train))

    #for n in range(2, 22):
    #    results.append(Train(55, 50, 0.02, 0, ResNet([64]*n, ResNet_Bottleneck), test, train))
    
    #for n in range(5, 22):
    #    results.append(Train(55, 50, 0.02, 0, ResNet([64]*n, SE_Basic), test, train))
    
    #for n in range(3, 22):
    #    results.append(Train(55, 50, 0.02, 0, ResNet([64]*n, SE_Bottleneck), test, train))
    
    print(results)
    input("Press Enter.")