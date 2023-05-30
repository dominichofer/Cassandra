import wx
from core import uint64, Position, possible_moves, play, first_set_field


class BoardModel:
    def __init__(self, pos: Position, crosses: uint64 = uint64(0), texts = None):
        self.pos = pos
        self.crosses = crosses
        if texts is None:
            self.clear_texts()
        else:
            self.texts = texts

    def clear_texts(self):
        self.texts = [None] * 64
        

class BoardView:
    def __init__(self, panel, size, model: BoardModel, black_and_white: bool):
        self.panel = panel
        self.size = size
        self.grid_offset = int(size / 2)
        self.total_size = 8 * self.size + self.grid_offset + 3
        self.line_width = 3
        self.model = model

        if black_and_white:
            self.background_color = wx.Colour("white")
            self.board_color = wx.Colour("white")
            self.grid_color = wx.Colour("black")
            self.player_fill_color = wx.Colour("black")
            self.player_border_color = wx.Colour("black")
            self.opponent_fill_color = wx.Colour("white")
            self.opponent_border_color = wx.Colour("black")
            self.cross_color = wx.Colour("black")
            self.text_color = wx.Colour("black")
        else:
            self.background_color = wx.Colour(171, 171, 171)
            self.board_color = wx.Colour(0, 100, 0)
            self.grid_color = wx.Colour(0, 50, 0)
            self.player_fill_color = wx.Colour("black")
            self.player_border_color = wx.Colour("black")
            self.opponent_fill_color = wx.Colour("white")
            self.opponent_border_color = wx.Colour("white")
            bordeaux = wx.Colour(95, 2, 31)
            self.cross_color = bordeaux
            self.text_color = bordeaux
        
    @staticmethod
    def board_to_mask(x, y) -> uint64:
        return uint64(1) << uint64(63 - x - 8 * y)
    
    def screen_to_mask(self, x, y) -> uint64:
        return self.board_to_mask(int((x - self.grid_offset) / self.size), int((y - self.grid_offset) / self.size))

    def refresh(self):
        self.panel.Refresh()
        
    def paint(self):
        dc = wx.BufferedPaintDC(self.panel)

        # Background
        dc.SetBrush(wx.Brush(self.background_color))
        dc.DrawRectangle(0, 0, self.total_size, self.total_size)

        # Rows / Cols
        dc.SetTextForeground(self.grid_color)
        font_size = int(self.size / 3)
        dc.SetFont(wx.Font(font_size, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        for row in range(8):
            dc.DrawText(
                str(row + 1),
                int(self.size / 4 - font_size / 3),
                row * self.size + int(self.size / 2 - font_size / 1.5) + self.grid_offset)
        for x, col in enumerate(['A','B','C','D','E','F','G','H']):
            dc.DrawText(
                col,
                x * self.size + int(self.size / 2 - font_size / 3) + self.grid_offset,
                int(self.size / 4 - font_size / 1.5))

        # Board
        dc.SetBrush(wx.Brush(self.board_color))
        dc.SetPen(wx.Pen(self.grid_color, self.line_width))
        for x in range(8):
            for y in range(8):
                dc.DrawRectangle(
                    x * self.size + self.grid_offset,
                    y * self.size + self.grid_offset,
                    self.size + 1,
                    self.size + 1)

        # Small dots
        dc.SetBrush(wx.Brush(self.grid_color))
        dc.SetPen(wx.Pen(self.grid_color))
        for x in [2,6]:
            for y in [2,6]:
                dc.DrawCircle(
                    x * self.size + self.grid_offset,
                    y * self.size + self.grid_offset,
                    int(1.5 * self.line_width))
        
        # player discs
        dc.SetBrush(wx.Brush(self.player_fill_color))
        dc.SetPen(wx.Pen(self.player_border_color, self.line_width))
        for x in range(8):
            for y in range(8):
                if self.model.pos.P & self.board_to_mask(x, y):
                    dc.DrawCircle(
                        int((x + 0.5) * self.size) + self.grid_offset,
                        int((y + 0.5) * self.size) + self.grid_offset,
                        int(self.size * 0.48 - self.line_width))

        # Opponent discs
        dc.SetBrush(wx.Brush(self.opponent_fill_color))
        dc.SetPen(wx.Pen(self.opponent_border_color, self.line_width))
        for x in range(8):
            for y in range(8):
                if self.model.pos.O & self.board_to_mask(x, y):
                    dc.DrawCircle(
                        int((x + 0.5) * self.size) + self.grid_offset,
                        int((y + 0.5) * self.size) + self.grid_offset,
                        int(self.size * 0.48 - self.line_width))

        # Crosses
        dc.SetBrush(wx.Brush(self.cross_color))
        pen = wx.Pen(self.cross_color, self.line_width)
        dc.SetPen(pen)
        for x in range(8):
            for y in range(8):
                if self.model.crosses & self.board_to_mask(x, y):
                    dc.DrawLine(
                        int((x + 0.4) * self.size) + self.grid_offset,
                        int((y + 0.4) * self.size) + self.grid_offset,
                        int((x + 0.6) * self.size) + self.grid_offset,
                        int((y + 0.6) * self.size) + self.grid_offset)
                    dc.DrawLine(
                        int((x + 0.4) * self.size) + self.grid_offset,
                        int((y + 0.6) * self.size) + self.grid_offset,
                        int((x + 0.6) * self.size) + self.grid_offset,
                        int((y + 0.4) * self.size) + self.grid_offset)
                    
        # Texts
        dc.SetTextForeground(self.text_color)
        for x in range(8):
            for y in range(8):
                text = self.model.texts[63 - x - 8 * y]
                if text is not None:
                    dc.DrawText(
                        text,
                        x * self.size + self.grid_offset,
                        y * self.size + self.grid_offset)

        
class SetUpBoardController:
    def __init__(self, model, view, refresh_callback):
        self.model = model
        self.model.crosses = possible_moves(self.model.pos)
        self.view = view
        self.refresh_callback = refresh_callback
        self.action = self.do_nothing
    
    def do_nothing(self, mask):
        pass
    
    def add_to_player(self, mask):
        self.model.pos.P |= mask
        self.model.pos.O &= ~mask
        self.model.crosses = possible_moves(self.model.pos)
    
    def add_to_opponent(self, mask):
        self.model.pos.P &= ~mask
        self.model.pos.O |= mask
        self.model.crosses = possible_moves(self.model.pos)
    
    def remove(self, mask):
        self.model.pos.P &= ~mask
        self.model.pos.O &= ~mask
        self.model.crosses = possible_moves(self.model.pos)
        
    def apply(self, mask):
        self.action(mask)

    def new_state(self, action):
        self.action = action

    def on_left_down(self, mask: uint64):
        if self.model.pos.P & mask:
            self.new_state(self.remove)
        else:
            self.new_state(self.add_to_player)
        self.on_move(mask)

    def on_right_down(self, mask: uint64):
        if self.model.pos.O & mask:
            self.new_state(self.remove)
        else:
            self.new_state(self.add_to_opponent)
        self.on_move(mask)

    def on_up(self, mask: uint64):
        self.new_state(self.do_nothing)

    def on_move(self, mask: uint64):
        self.apply(mask)
        self.refresh_view()

    def refresh_view(self):
        self.view.refresh()
        self.refresh_callback()

            
class BoardPanel(wx.Panel):
    def __init__(self, parent, size, pos: Position, refresh_callback, black_and_white: bool = False):
        self.model = BoardModel(pos)
        self.view = BoardView(self, size, self.model, black_and_white)
        super(BoardPanel, self).__init__(parent, size=(self.view.total_size, self.view.total_size), style=wx.SIMPLE_BORDER)
        self.controller = SetUpBoardController(self.model, self.view, refresh_callback)

        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_right_down)
        self.Bind(wx.EVT_LEFT_UP, self.on_up)
        self.Bind(wx.EVT_RIGHT_UP, self.on_up)
        self.Bind(wx.EVT_MOTION, self.on_move)

    @property
    def pos(self):
        return self.model.pos

    @pos.setter
    def pos(self, pos: Position):
        self.model.pos = pos
        self.model.crosses = possible_moves(self.model.pos)
        self.model.clear_texts()
        self.controller.refresh_view()

    def on_paint(self, event):
        self.view.paint()

    def on_erase(self, event):
        pass

    def on_left_down(self, event):
        mask = self.view.screen_to_mask(event.x, event.y)
        self.controller.on_left_down(mask)

    def on_right_down(self, event):
        mask = self.view.screen_to_mask(event.x, event.y)
        self.controller.on_right_down(mask)

    def on_up(self, event):
        mask = self.view.screen_to_mask(event.x, event.y)
        self.controller.on_up(mask)

    def on_move(self, event):
        mask = self.view.screen_to_mask(event.x, event.y)
        self.controller.on_move(mask)
