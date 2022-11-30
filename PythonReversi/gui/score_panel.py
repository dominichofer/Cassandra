from copy import deepcopy
from dataclasses import dataclass
from collections import defaultdict
import wx

@dataclass
class WhiskedValue:
    upper: float
    mid: float
    lower: float


class Serie:
    def __init__(self):
        self.xs = []
        self.ys = []

    def add(self, x: int, y: WhiskedValue):
        self.xs.append(x)
        self.ys.append(y)

    def __iter__(self):
        return zip(self.xs, self.ys)
        

class ScorePanel(wx.Panel):
    def __init__(self, parent, width, height):
        super(ScorePanel, self).__init__(parent, size=(width, height), style=wx.SIMPLE_BORDER)
        self.width = width
        self.height = height
        self.background_color = wx.Colour(10, 10, 10)
        self.grid_color = wx.Colour(60, 60, 60)
        self.middle_line_color = wx.Colour(150, 150, 150)
        self.serie_color = wx.Colour(200, 0, 0)
        self.series = defaultdict(Serie)

        self.Bind(wx.EVT_PAINT, self.on_paint)

    def __point(self, depth, score):
        return wx.Point(int(depth / 60 * self.width), int((1 - (score + 32) / 65) * self.height))

    def add(self, name: str, x: int, y):
        if not isinstance(y, WhiskedValue):
            y = WhiskedValue(y, y, y)
        self.series[name].add(x, y)

    def clear(self):
        self.series.clear()

    def on_paint(self, event):
        dc = wx.BufferedPaintDC(self)
        dc.SetBackground(wx.Brush(self.background_color))
        dc.Clear()

        # Grid
        dc.SetPen(wx.Pen(self.grid_color))
        for depth in range(0, 61, 10):
            dc.DrawLine(self.__point(depth, -32), self.__point(depth, 32))
        for score in range(-32, 33, 8):
            dc.DrawLine(self.__point(0, score), self.__point(60, score))
        
        # Zero line
        dc.SetPen(wx.Pen(self.middle_line_color))
        dc.DrawLine(self.__point(0, 0), self.__point(60, 0))

        # Confidence interval
        dimm = wx.Colour(int(self.serie_color.red / 2), int(self.serie_color.green / 2), int(self.serie_color.blue / 2))
        dc.SetPen(wx.Pen(self.serie_color))
        #dc.SetBrush(wx.Brush(self.serie_color))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        for serie in self.series.values():
            for depth, score in serie:
                dc.DrawCircle(self.__point(depth, score.mid), 2)
                dc.DrawLine(self.__point(depth, score.lower), self.__point(depth, score.upper))
            #dc.DrawLines([self.__point(depth, score.upper) for depth, score in serie])
            #dc.DrawLines([self.__point(depth, score.mid) for depth, score in serie])
            #dc.DrawLines([self.__point(depth, score.lower) for depth, score in serie])