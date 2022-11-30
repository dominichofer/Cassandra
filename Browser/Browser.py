import wx
import numpy
import numpy.random
import time
import math
import matplotlib.pyplot as plt
import struct
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torchvision as tv
from PIL import Image, ImageDraw
import sys

class BitBoard:
    def __init__(self, b:numpy.uint64=0):
        self.__b = b

    @staticmethod
    def FromField(x, y):
        return BitBoard(numpy.uint64(1) << (y * 8 + x))

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
    P = BitBoard()
    O = BitBoard()
    def __init__(self, P, O):
        if not isinstance(P, BitBoard):
            P = BitBoard(P)
        if not isinstance(O, BitBoard):
            O = BitBoard(O)
        self.P = P
        self.O = O

    @staticmethod
    def Start():
        return Position(BitBoard(0x0000000810000000), BitBoard(0x0000001008000000))

def PlayPass(pos:Position) -> Position:
    return Position(pos.O, pos.P)

def Play(pos:Position, x, y) -> Position:
    flips = Flips(pos, x, y)
    return Position(pos.O ^ flips, pos.P ^ flips ^ BitBoard.FromField(x, y))

def FlipsInOneDirection(pos:Position, x, y, dx, dy) -> BitBoard:
    flips = BitBoard()
    x += dx
    y += dy
    while (x >= 0) and (x < 8) and (y >= 0) and (y < 8):
        if pos.O[x,y]:
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
    maskO = pos.O & BitBoard(0x7E7E7E7E7E7E7E7E)
    
    flip1 = maskO & (pos.P << 1);
    flip2 = maskO & (pos.P >> 1);
    flip3 = pos.O & (pos.P << 8);
    flip4 = pos.O & (pos.P >> 8);
    flip5 = maskO & (pos.P << 7);
    flip6 = maskO & (pos.P >> 7);
    flip7 = maskO & (pos.P << 9);
    flip8 = maskO & (pos.P >> 9);

    flip1 |= maskO & (flip1 << 1);
    flip2 |= maskO & (flip2 >> 1);
    flip3 |= pos.O & (flip3 << 8);
    flip4 |= pos.O & (flip4 >> 8);
    flip5 |= maskO & (flip5 << 7);
    flip6 |= maskO & (flip6 >> 7);
    flip7 |= maskO & (flip7 << 9);
    flip8 |= maskO & (flip8 >> 9);

    mask1 = maskO & (maskO << 1);
    mask2 = mask1 >> 1;
    mask3 = pos.O & (pos.O << 8);
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

    return ~(pos.P | pos.O) & (flip1 | flip2 | flip3 | flip4 | flip5 | flip6 | flip7 | flip8);


class State():
    def __init__(self, pos:Position):
        self.pos = pos
        self.bb = BitBoard()
    def add(self, x, y):
        if not self.bb[x,y]:
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
        super(PositionFrame, self).__init__(None, title='Browser', style=wx.DEFAULT_FRAME_STYLE & ~wx.RESIZE_BORDER)

        self.TextBox = wx.TextCtrl(self, size=(403,-1), style = wx.TE_PROCESS_ENTER)
        self.Bind(wx.EVT_TEXT_ENTER, self.TextEnter)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(DrawingPanel(self, pos))
        vbox.Add(self.TextBox)
        self.SetSizer(vbox)
        self.Fit()
        self.Show()


    def TextEnter(self, event):
        wx.MessageBox(self.TextBox.GetValue())

class DrawingPanel(wx.Panel):
    def __init__(self, parent, pos:Position):
        self.d = 50
        self.pos = pos
        self.state = DoNothing(self.pos)

        super(DrawingPanel, self).__init__(parent, size=(8 * self.d + 3, 8 * self.d + 3), style=wx.SIMPLE_BORDER)
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
        dc.DrawCircle(int((x + 0.5) * self.d), int((y + 0.5) * self.d), int(self.d * 0.45))
        
    def DrawX(self, dc:wx.BufferedPaintDC, colour:wx.Colour, x, y):
        pen = wx.Pen(colour)
        pen.SetWidth(int(self.d/20))
        dc.SetPen(pen)
        dc.SetBrush(wx.Brush(colour))
        dc.DrawLine(int((x + 0.4) * self.d), int((y + 0.4) * self.d), int((x + 0.6) * self.d), int((y + 0.6) * self.d))
        dc.DrawLine(int((x + 0.4) * self.d), int((y + 0.6) * self.d), int((x + 0.6) * self.d), int((y + 0.4) * self.d))

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
                dc.DrawCircle(x * self.d, y * self.d, int(self.d * 0.07))

        possible_moves = PossibleMoves(self.pos)   
        for x in range(8):
            for y in range(8):
                if self.pos.P[x,y]:
                    self.DrawDisc(dc, wx.Colour("black"), x, y)
                elif self.pos.O[x,y]:
                    self.DrawDisc(dc, wx.Colour("white"), x, y)
                elif possible_moves[x,y]:
                    self.DrawX(dc, wx.Colour("red"), x, y)

def to_png(pos):
    d = 10
    out = Image.new("RGB", (8*d+1,8*d+1), (0,0,0))
    draw = ImageDraw.Draw(out)

    # Board
    for x in range(8):
        for y in range(8):
            draw.rectangle([x*d, y*d, (x+1)*d, (y+1)*d], (0,100,0), (0,50,0))
    # Small black dots
    for x in [2,6]:
        for y in [2,6]:
            draw.ellipse([x*d-0.05*d, y*d-0.05*d, x*d+0.05*d+1, y*d+0.05*d+1], (0,50,0), (0,50,0))

    possible_moves = PossibleMoves(pos)
    for x in range(8):
        for y in range(8):
            if pos.P[x,y]:
                draw.ellipse([x*d+0.075*d+1, y*d+0.075*d+1, (x+1)*d-0.075*d, (y+1)*d-0.075*d], (0,0,0), (0,0,0))
            elif pos.O[x,y]:
                draw.ellipse([x*d+0.075*d+1, y*d+0.075*d+1, (x+1)*d-0.075*d, (y+1)*d-0.075*d], (255,255,255), (255,255,255))
            elif possible_moves[x,y]:
                draw.line([x*d+0.4*d, y*d+0.4*d, (x+1)*d-0.4*d, (y+1)*d-0.4*d], (255,0,0), int(d/20))
                draw.line([x*d+0.4*d, (y+1)*d-0.4*d, (x+1)*d-0.4*d, y*d+0.4*d], (255,0,0), int(d/20))
                
    name = ''
    for y in range(8):
        for x in range(8):
            if pos.P[x,y]:
                name += 'X'
            elif pos.O[x,y]:
                name += 'O'
            else:
                name += '-'

    out.save(f'G:\\Reversi\\perft\\ply7\\{name}.png')

def DataGenerator():
    chunk_size = struct.calcsize('<QQQQ')
    with open(f'G:\\Reversi\\perft\\perft21_ply7.pos', "rb") as file:
        while True:
            try:
                P, O, *_ = struct.unpack('<QQQQ', file.read(chunk_size))
                yield Position(P, O)
            except:
                break

if __name__ == '__main__':
    #to_png(Position(BitBoard(0x0000000810000000), BitBoard(0x0000001008000000)))
    #name = ''
    #for x in range(8):
    #    for y in range(8):
    #        if pos.P[x,y]:
    #            name += 'X'
    #        elif pos.O[x,y]:
    #            name += 'O'
    #        else:
    #            name += '-'
    #print(name)
    #for pos in DataGenerator():
    #    to_png(pos)

    app = wx.App()
    frame = PositionFrame(Position.start())
    app.MainLoop()