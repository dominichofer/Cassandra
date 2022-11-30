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
    def __init__(self, panel, size, model: BoardModel):
        self.panel = panel
        self.size = size
        self.model = model
        self.grid_color = wx.Colour(0, 50, 0)
        self.background_color = wx.Colour(0, 150, 0)
        self.player_color = wx.Colour("black")
        self.opponent_color = wx.Colour("white")
        self.crosses_color = wx.Colour("red")
        self.text_color = wx.Colour("red")
        
    @staticmethod
    def board_to_mask(x, y) -> uint64:
        return uint64(1) << uint64(63 - x - 8 * y)
    
    def screen_to_mask(self, x, y) -> uint64:
        return self.board_to_mask(int(x / self.size), int(y / self.size))

    def refresh(self):
        self.panel.Refresh()
        
    def paint(self):
        dc = wx.BufferedPaintDC(self.panel)

        # Board
        dc.SetPen(wx.Pen(self.grid_color))
        dc.SetBrush(wx.Brush(self.background_color))
        for x in range(8):
            for y in range(8):
                dc.DrawRectangle(x * self.size, y * self.size, self.size + 1, self.size + 1)

        # Small black dots
        dc.SetPen(wx.Pen(self.grid_color))
        dc.SetBrush(wx.Brush(self.grid_color))
        for x in [2,6]:
            for y in [2,6]:
                dc.DrawCircle(x * self.size, y * self.size, int(self.size * 0.07))
        
        # player discs
        dc.SetPen(wx.Pen(self.player_color))
        dc.SetBrush(wx.Brush(self.player_color))
        for x in range(8):
            for y in range(8):
                if self.model.pos.P & self.board_to_mask(x, y):
                    dc.DrawCircle(int((x + 0.5) * self.size), int((y + 0.5) * self.size), int(self.size * 0.45))

        # Opponent discs
        dc.SetPen(wx.Pen(self.opponent_color))
        dc.SetBrush(wx.Brush(self.opponent_color))
        for x in range(8):
            for y in range(8):
                if self.model.pos.O & self.board_to_mask(x, y):
                    dc.DrawCircle(int((x + 0.5) * self.size), int((y + 0.5) * self.size), int(self.size * 0.45))

        # Crosses
        pen = wx.Pen(self.crosses_color)
        pen.SetWidth(int(self.size * 0.05))
        dc.SetPen(pen)
        dc.SetBrush(wx.Brush(self.crosses_color))
        for x in range(8):
            for y in range(8):
                if self.model.crosses & self.board_to_mask(x, y):
                    dc.DrawLine(int((x + 0.4) * self.size), int((y + 0.4) * self.size), int((x + 0.6) * self.size), int((y + 0.6) * self.size))
                    dc.DrawLine(int((x + 0.4) * self.size), int((y + 0.6) * self.size), int((x + 0.6) * self.size), int((y + 0.4) * self.size))
                    
        # Texts
        dc.SetTextForeground(self.text_color)
        for x in range(8):
            for y in range(8):
                text = self.model.texts[63 - x - 8 * y]
                if text is not None:
                    dc.DrawText(text, x * self.size, y * self.size)


        
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
    def __init__(self, parent, size, pos: Position, refresh_callback):
        super(BoardPanel, self).__init__(parent, size=(8 * size + 3, 8 * size + 3), style=wx.SIMPLE_BORDER)
        self.model = BoardModel(pos)
        self.view = BoardView(self, size, self.model)
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
