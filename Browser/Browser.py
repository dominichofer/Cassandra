import wx
import numpy
import numpy.random
import time
import math
import matplotlib.pyplot as plt
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader, TensorDataset

class BitBoard:
    def __init__(self, b:numpy.uint64=0):
        self.__b = b

    def __eq__(self, o):
        return self.__b == o.__b
    def __neq__(self, o):
        return self.__b != o.__b

    def __and__(self, o):
        return BitBoard(self.__b & o.__b)
    def __or__(self, o):
        return BitBoard(self.__b | o.__b)
    def __invert__(self):
        return BitBoard(~self.__b)

    def __lshift__(self, i):
        return BitBoard(self.__b << i)
    def __rshift__(self, i):
        return BitBoard(self.__b >> i)

    def __getitem__(self, key) -> bool:
        x,y = key
        return bool(self.__b & (1 << (63 - x - 8 * y)))
    def __setitem__(self, key, value:bool):
        x,y = key
        if value:
            self.__b |= (1 << (63 - x - 8 * y))
        else:
            self.__b &= ~(1 << (63 - x - 8 * y))

class Position:
    def __init__(self, P=BitBoard(), O=BitBoard()):
        if not isinstance(P, BitBoard):
            P = BitBoard(P)
        if not isinstance(O, BitBoard):
            O = BitBoard(O)
        self.P = P
        self.O = O

def PlayPass(pos:Position):
    return Position(pos.Opponent(), pos.P)

def FlipsInOneDirection(pos:Position, x, y, dx, dy) -> BitBoard:
    flips = BitBoard()
    x += dx
    y += dy
    while (x >= 0) and (x < 8) and (y >= 0) and (y < 8):
        if pos.Opponent()[x,y]:
            flips[x,y] = True
        elif pos.P[x,y]:
            return flips
        else:
            break
        x += dx
        y += dy
    return BitBoard()

def Flips(pos:Position, x, y) -> BitBoard:
    return FlipsInOneDirection(pos, x, y, -1, -1) \
         | FlipsInOneDirection(pos, x, y, -1, +0) \
         | FlipsInOneDirection(pos, x, y, -1, +1) \
         | FlipsInOneDirection(pos, x, y, +0, -1) \
         | FlipsInOneDirection(pos, x, y, +0, +1) \
         | FlipsInOneDirection(pos, x, y, +1, -1) \
         | FlipsInOneDirection(pos, x, y, +1, +0) \
         | FlipsInOneDirection(pos, x, y, +1, +1)

def PossibleMoves(pos:Position) -> BitBoard:
    maskO = pos.Opponent() & BitBoard(0x7E7E7E7E7E7E7E7E)
    
    flip1 = maskO & (pos.P << 1);
    flip2 = maskO & (pos.P >> 1);
    flip3 = pos.Opponent() & (pos.P << 8);
    flip4 = pos.Opponent() & (pos.P >> 8);
    flip5 = maskO & (pos.P << 7);
    flip6 = maskO & (pos.P >> 7);
    flip7 = maskO & (pos.P << 9);
    flip8 = maskO & (pos.P >> 9);

    flip1 |= maskO & (flip1 << 1);
    flip2 |= maskO & (flip2 >> 1);
    flip3 |= pos.Opponent() & (flip3 << 8);
    flip4 |= pos.Opponent() & (flip4 >> 8);
    flip5 |= maskO & (flip5 << 7);
    flip6 |= maskO & (flip6 >> 7);
    flip7 |= maskO & (flip7 << 9);
    flip8 |= maskO & (flip8 >> 9);

    mask1 = maskO & (maskO << 1);
    mask2 = mask1 >> 1;
    mask3 = pos.Opponent() & (pos.Opponent() << 8);
    mask4 = mask3 >> 8;
    mask5 = maskO & (maskO << 7);
    mask6 = mask5 >> 7;
    mask7 = maskO & (maskO << 9);
    mask8 = mask7 >> 9;

    flip1 |= mask1 & (flip1 << 2);
    flip2 |= mask2 & (flip2 >> 2);
    flip3 |= mask3 & (flip3 << 16);
    flip4 |= mask4 & (flip4 >> 16);
    flip5 |= mask5 & (flip5 << 14);
    flip6 |= mask6 & (flip6 >> 14);
    flip7 |= mask7 & (flip7 << 18);
    flip8 |= mask8 & (flip8 >> 18);

    flip1 |= mask1 & (flip1 << 2);
    flip2 |= mask2 & (flip2 >> 2);
    flip3 |= mask3 & (flip3 << 16);
    flip4 |= mask4 & (flip4 >> 16);
    flip5 |= mask5 & (flip5 << 14);
    flip6 |= mask6 & (flip6 >> 14);
    flip7 |= mask7 & (flip7 << 18);
    flip8 |= mask8 & (flip8 >> 18);

    flip1 <<= 1;
    flip2 >>= 1;
    flip3 <<= 8;
    flip4 >>= 8;
    flip5 <<= 7;
    flip6 >>= 7;
    flip7 <<= 9;
    flip8 >>= 9;

    return ~(pos.P | pos.Opponent()) & (flip1 | flip2 | flip3 | flip4 | flip5 | flip6 | flip7 | flip8);


class State():
    def __init__(self, pos:Position):
        self.pos = pos
        self.bb = BitBoard()
    def add(self, x, y):
        if self.bb[x,y] == False:
            self.bb[x,y] = True
            return True
        return False
        
class DoNothing(State):
    def __init__(self, pos:Position):
        State.__init__(self, pos)
    def add(self, x, y):
        return False
    def get(self):
        return self.pos

class AddPlayer(State):
    def __init__(self, pos:Position):
        State.__init__(self, pos)
    def get(self):
        self.pos.P |= self.bb
        self.pos.O &= ~self.bb
        return self.pos

class AddOpponent(State):
    def __init__(self, pos:Position):
        State.__init__(self, pos)
    def get(self):
        self.pos.P &= ~self.bb
        self.pos.O |= self.bb
        return self.pos

class Remove(State):
    def __init__(self, pos:Position):
        State.__init__(self, pos)
    def get(self):
        self.pos.P &= ~self.bb
        self.pos.O &= ~self.bb
        return self.pos
  

class PositionFrame(wx.Frame):
    def __init__(self, pos:Position):
        wx.Frame.__init__(self, None, title='Browser', style=wx.DEFAULT_FRAME_STYLE & ~wx.RESIZE_BORDER)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(DrawingPanel(self, pos))
        self.SetSizer(sizer)
        self.Fit()
        self.Show(True)

class DrawingPanel(wx.Panel):
    def __init__(self, parent, pos:Position):
        self.d = 50
        self.pos = pos
        self.state = DoNothing(self.pos)
        self.toggle_bb = BitBoard()
        wx.Panel.__init__(self, parent, size=(8 * self.d + 3, 8 * self.d + 3), style=wx.SIMPLE_BORDER)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_RIGHT_DOWN, self.OnRightDown)
        self.Bind(wx.EVT_MOTION, self.OnMove)
        self.Bind(wx.EVT_LEFT_UP, self.OnUp)
        self.Bind(wx.EVT_RIGHT_UP, self.OnUp)

    def OnLeftDown(self, event):
        x,y = int(event.x / self.d), int(event.y / self.d)
        if self.pos.P[x,y]:
            self.state = Remove(self.pos)
        else:
            self.state = AddPlayer(self.pos)
        self.state.add(x,y)
        self.pos = self.state.get()
        self.Refresh()

    def OnRightDown(self, event):
        x,y = int(event.x / self.d), int(event.y / self.d)
        if self.pos.O[x,y]:
            self.state = Remove(self.pos)
        else:
            self.state = AddOpponent(self.pos)
        self.state.add(x,y)
        self.pos = self.state.get()
        self.Refresh()

    def OnMove(self, event):
        x,y = int(event.x / self.d), int(event.y / self.d)
        if self.state.add(x,y):
            self.pos = self.state.get()
            self.Refresh()

    def OnUp(self, event):
        self.pos = self.state.get()
        self.state = DoNothing(self.pos)
        self.Refresh()
 
    def DrawDisc(self, dc, colour, x, y):
        dc.SetPen(wx.Pen(colour))
        dc.SetBrush(wx.Brush(colour))
        dc.DrawCircle((x + 0.5) * self.d, (y + 0.5) * self.d, self.d * 0.45)
        
    def DrawX(self, dc:wx.BufferedPaintDC, colour:wx.Colour, x, y):
        pen = wx.Pen(colour)
        pen.SetWidth(self.d/20)
        dc.SetPen(pen)
        dc.SetBrush(wx.Brush(colour))
        dc.DrawLine((x + 0.4) * self.d, (y + 0.4) * self.d, (x + 0.6) * self.d, (y + 0.6) * self.d)
        dc.DrawLine((x + 0.4) * self.d, (y + 0.6) * self.d, (x + 0.6) * self.d, (y + 0.4) * self.d)

    def OnPaint(self, event):
        dc = wx.BufferedPaintDC(self)

        grid = wx.Colour(0,50,0)
        background = wx.Colour(0,100,0)

        dc.SetPen(wx.Pen(grid))
        # Board
        for x in range(8):
            for y in range(8):
                dc.SetBrush(wx.Brush(background))
                dc.DrawRectangle(x * self.d, y * self.d, self.d + 1, self.d + 1)
        # Small black dots
        for x in [2,6]:
            for y in [2,6]:
                dc.SetBrush(wx.Brush(grid))
                dc.DrawCircle(x * self.d, y * self.d, self.d * 0.07)

        possible_moves = PossibleMoves(self.pos)   
        for x in range(8):
            for y in range(8):
                if self.pos.P[x,y]:
                    self.DrawDisc(dc, wx.Colour("black"), x, y)
                elif self.pos.O[x,y]:
                    self.DrawDisc(dc, wx.Colour("white"), x, y)
                elif possible_moves[x,y]:
                    self.DrawX(dc, wx.Colour("red"), x, y)

def HeatMap(x,y):
    r_squared = numpy.corrcoef(x, y)[0,1]**2
    std = numpy.std(x - y)

    heatmap, xedges, yedges = numpy.histogram2d(x, y, bins=(65, 129), range=((-64,64),(-64,64)))

    plt.text(35, -55, '$R^2$={:.3f}\n$\sigma$={:.3f}'.format(r_squared, std))
    plt.set_cmap('gist_heat_r')
    plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower')
    plt.show()
    
class ConvBn2d(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, padding):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_size)

    def forward(self, x):
        return self.bn(self.conv(x))

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, padding):
        super(ConvBnRelu2d, self).__init__()
        self.conv_bn = ConvBn2d(in_size, out_size, kernel_size, padding)

    def forward(self, x):
        return F.relu(self.conv_bn(x), inplace=True)

class Res1(nn.Module):
    def __init__(self, external, internal):
        super(Res1, self).__init__()
        self.layer1 = ConvBnRelu2d(internal, internal, 3, 1) # 3x3
        self.layer2 = ConvBnRelu2d(internal, internal, 3, 1) # 3x3

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        return F.relu(x + y, inplace=True)

class Res2(nn.Module):
    def __init__(self, external, internal):
        super(Res2, self).__init__()
        self.layer1 = ConvBnRelu2d(external, internal, 1, 0) # 1x1
        self.layer2 = ConvBnRelu2d(internal, internal, 3, 1) # 3x3
        self.layer3 = ConvBn2d(internal, external, 1, 0) # 1x1

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        return F.relu(x + y, inplace=True)

#class Res2(nn.Module):
#    def __init__(self, external, internal):
#        super(Res2, self).__init__()
#        self.layer1 = ConvBnRelu2d(external, internal, 1, 0) # 1x1
#        self.layer2 = ConvBnRelu2d(internal, internal, 3, 1) # 3x3
#        self.layer3 = ConvBn2d(internal, external, 1, 0) # 1x1

#    def forward(self, x):
#        y = self.layer1(x)
#        y = self.layer2(y)
#        y = self.layer3(y)
#        return F.relu(x + y, inplace=True)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ValueHead(nn.Module):
    def __init__(self, external, internal):
        super(ValueHead, self).__init__()
        self.layer1 = ConvBnRelu2d(external, 1, 1, 0)
        self.full = nn.Linear(internal, internal)
        self.shrink = nn.Linear(internal, 1)

    def forward(self, x):
        y = self.layer1(x)
        y = F.relu(self.full(y.flatten(1, -1)), inplace=True)
        return F.hardtanh(self.shrink(y), -64, 64, inplace=True)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def ResNet(skip_channels, internal_channels:list):
    # has 3*depth+3 hidden layers
    return nn.Sequential(
        ConvBnRelu2d(2, skip_channels, 3, 1), # 3x3
        *[SEBottleneck(skip_channels, ic) for ic in internal_channels],
        ValueHead(skip_channels, 64)
        )

def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = numpy.prod(param.size())
            #if param.dim() > 1:
            #    print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            #else:
            #    print(name, ':', num_param)
            total_param += num_param
    return total_param

class Drawer:
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

def Data(e):
    with open("G:\\Reversi\\rnd\\e{}.psc".format(e), "rb") as file:
        for i in range(200_000):
        #while True:
            try:
                P, O, sc = struct.unpack('<QQb', file.read(struct.calcsize('<QQb')))
                yield Position(P, O), sc
            except:
                break

def to_list(pos:Position):
    return [1 if pos.P[i, j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.Opponent()[i, j] else 0 for i in range(8) for j in range(8)]

def to_list_8(pos:Position):
    return [1 if pos.P[i, j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.Opponent()[i, j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.P[i, 7-j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.Opponent()[i, 7-j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.P[7-i, j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.Opponent()[7-i, j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.P[7-i, 7-j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.Opponent()[7-i, 7-j] else 0 for i in range(8) for j in range(8)] \
         + [1 if pos.P[i, j] else 0 for j in range(8) for i in range(8)] \
         + [1 if pos.Opponent()[i, j] else 0 for j in range(8) for i in range(8)] \
         + [1 if pos.P[i, 7-j] else 0 for j in range(8) for i in range(8)] \
         + [1 if pos.Opponent()[i, 7-j] else 0 for j in range(8) for i in range(8)] \
         + [1 if pos.P[7-i, j] else 0 for j in range(8) for i in range(8)] \
         + [1 if pos.Opponent()[7-i, j] else 0 for j in range(8) for i in range(8)] \
         + [1 if pos.P[7-i, 7-j] else 0 for j in range(8) for i in range(8)] \
         + [1 if pos.Opponent()[7-i, 7-j] else 0 for j in range(8) for i in range(8)]

def GetDataTensor():
    test_pos = []
    test_sc = []
    train_pos = []
    train_sc = []
    for e in range(18,23):
        for index, (pos, sc) in enumerate(Data(e)):
            if index < 50_000:
                test_pos = torch.cat((test_pos, torch.tensor(to_list(pos), dtype=torch.float)), 0)
                test_sc = torch.cat((test_sc, torch.tensor([sc], dtype=torch.float)), 0)
            else:
                train_pos = torch.cat((train_pos, torch.tensor(to_list_8(pos), dtype=torch.float)), 0)
                train_sc = torch.cat((train_sc, torch.tensor([sc]*8, dtype=torch.float)), 0)
    return test_pos, test_sc, train_pos, train_sc

def GetData():
    train = []
    test = []
    for e in range(18, 23):
        for index, data in enumerate(Data(e)):
            if index < 50_000:
                test.append(data)
            else:
                train.append(data)
    return train, test

def collate_fn(data):
    pos, sc = zip(*data)
    pos = torch.tensor([to_list(x) for x in pos], dtype=torch.float).view(-1, 2, 8, 8)
    sc = torch.tensor(sc, dtype=torch.float)
    return pos, sc

def collate_fn_8(data):
    pos, sc = zip(*data)
    pos = torch.tensor([to_list_8(x) for x in pos], dtype=torch.float).view(-1, 2, 8, 8)
    sc = torch.tensor([x for x in sc for _ in range(8)], dtype=torch.float)
    return pos, sc

if __name__ == '__main__':
    #import random
    #device = torch.device("cuda")
    #model = nn.DataParallel(ResNet(64, [64]*9)).to(device)
    ##model.load_state_dict(torch.load("G:\\Reversi\\ResNet_64_64_9.w"))
    #model.eval()

    #for i in range(100):
    #    x = [to_list(Position(random.getrandbits(64), random.getrandbits(64))) for _ in range(32)]
    #    x = torch.tensor(x, dtype=torch.float, requires_grad=False).view(-1, 3, 8, 8)
    #    start = time.time()
    #    model(x.to(device))
    #    end = time.time()
    #    print(end-start)

    drawer = Drawer()
    device = torch.device("cuda")

    train, test = GetData()
    train = collate_fn_8(train)
    test = collate_fn(test)
    #test_pos, test_sc, train_pos, train_sc = GetDataTensor()
    print('DataLoaded')
    
    test_size = len(test[0])
    train_size = len(train[0])
    test_loader = DataLoader(TensorDataset(*test), batch_size=8*1024, shuffle=True, pin_memory=True)
    train_loader = DataLoader(TensorDataset(*train), batch_size=8*1024, shuffle=True, pin_memory=True)
    loss_fn = nn.MSELoss()

    for depth in range(9,10):
        model = ResNet(64, [64]*depth)
        #model.eval()
        #start = time.time()
        #for i in range(1000):
        #    model(test[0][i].view(-1, 2, 8, 8))
        #stop = time.time()
        #model.train()
        #print("CPU inference time: ", stop-start)
        model = nn.DataParallel(model).to(device)
        #model.load_state_dict(torch.load("G:\\Reversi\\SE9.w"))
        print('number of trainable parameters =', count_parameters(model))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

        for epoch in range(22):
            start = time.time()
            train_loss = 0
            for x, y in train_loader:
                #x = torch.tensor(to_list(x), dtype=torch.float).view(x.size()[0], 2, 8, 8).to(device)
                #y = torch.tensor(y, dtype=torch.float).to(device)
                loss = loss_fn(model(x.to(device)).view(-1), y.to(device))
                train_loss += loss.item() * x.size()[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss = math.sqrt(train_loss / train_size)
            end = time.time()

            with torch.no_grad():
                model.eval()
                test_loss = 0
                for x, y in test_loader:
                    #x = torch.tensor(to_list(x), dtype=torch.float, requires_grad=False).view(x.size()[0], 2, 8, 8).to(device)
                    #y = torch.tensor(y, dtype=torch.float, requires_grad=False).to(device)
                    loss = loss_fn(model(x.to(device)).view(-1), y.to(device))
                    test_loss += loss.item() * x.size()[0]
                test_loss = math.sqrt(test_loss / test_size)
                model.train()

                drawer.add(train_loss, test_loss)
                print(epoch, train_loss, test_loss, end - start)

            scheduler.step()
            torch.save(model.state_dict(), "G:\\Reversi\\SE9.w")
                #HeatMap(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy())

    #with open("G:\\Reversi\\rnd\\e1.psc", "rb") as file:
    #    file.seek(100000*struct.calcsize('<QQb'))
    #    data = struct.unpack('<QQ', file.read(struct.calcsize('<QQ')))

    #app = wx.App()
    #frame = PositionFrame(Position(BitBoard(data[0]), BitBoard(data[1])))
    ##frame = PositionFrame(Position(BitBoard(0x0000001800000000), BitBoard(0x0000000018000000)))
    #app.MainLoop()