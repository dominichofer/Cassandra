import wx
import wx.grid
from pathlib import Path
from core import Position, possible_moves
from gui import BoardPanel, ScorePanel, WhiskedValue
import edax
import db


class PositionFrame(wx.Frame):

    def __init__(self):
        super(PositionFrame, self).__init__(None, title='Position', style=wx.DEFAULT_FRAME_STYLE & ~wx.RESIZE_BORDER)

        self.board = BoardPanel(self, 50, Position.start(), self.pos_update)

        self.pos_text = wx.TextCtrl(self, size=(483, -1), style=wx.TE_PROCESS_ENTER)
        self.pos_text.SetFont(wx.Font(10, wx.MODERN, wx.NORMAL, wx.NORMAL, faceName='Consolas'))
        self.Bind(wx.EVT_TEXT_ENTER, self.text_enter)
        
        player = wx.StaticText(self, label='player: ')
        opponent = wx.StaticText(self, label='Opponent: ')
        empty_count = wx.StaticText(self, label='Empties: ')
        moves = wx.StaticText(self, label='Moves: ')
        self.player = wx.StaticText(self, label='02')
        self.opponent = wx.StaticText(self, label='02')
        self.empty_count = wx.StaticText(self, label='60')
        self.moves = wx.StaticText(self, label='04')
        
        self.button_start = wx.Button(self, size=(40, -1), id=0, label='start')
        self.button_clear = wx.Button(self, size=(40, -1), id=1, label='clear')
        self.button_flip = wx.Button(self, size=(40, -1), id=2, label='flip')
        self.Bind(wx.EVT_BUTTON, self.on_button_start, id=0)
        self.Bind(wx.EVT_BUTTON, self.on_button_clear, id=1)
        self.Bind(wx.EVT_BUTTON, self.on_button_flip, id=2)

        self.score = ScorePanel(self, 340, 160)
        self.button_eval = wx.Button(self, size=(40, -1), id=3, label='eval')
        self.Bind(wx.EVT_BUTTON, self.on_button_eval, id=3)

        stats_box = wx.GridBagSizer(0, 0)
        stats_box.Add(player, pos=(0, 0), flag=wx.ALIGN_RIGHT)
        stats_box.Add(opponent, pos=(1, 0), flag=wx.ALIGN_RIGHT)
        stats_box.Add(empty_count, pos=(2, 0), flag=wx.ALIGN_RIGHT)
        stats_box.Add(moves, pos=(3, 0), flag=wx.ALIGN_RIGHT)
        stats_box.Add(self.player, pos=(0, 1))
        stats_box.Add(self.opponent, pos=(1, 1))
        stats_box.Add(self.empty_count, pos=(2, 1))
        stats_box.Add(self.moves, pos=(3, 1))

        pos_buttons = wx.GridSizer(1)
        pos_buttons.Add(self.button_start)
        pos_buttons.Add(self.button_clear)
        pos_buttons.Add(self.button_flip)

        eval_grid = wx.GridBagSizer(0, 0)
        eval_grid.Add(self.score, pos=(0, 0))
        eval_grid.Add(self.button_eval, pos=(1, 0), flag=wx.ALIGN_RIGHT)
        
        grid = wx.GridBagSizer(0, 0)
        grid.Add(self.board, pos=(0, 0), span=(3, 1))
        grid.Add(stats_box, pos=(0, 1), border=5, flag=wx.ALL)
        grid.Add((0, 238), pos=(1, 1))
        grid.Add(pos_buttons, pos=(2, 1), flag=wx.ALIGN_LEFT)
        grid.Add(eval_grid, pos=(0, 2), span=(2, 1))
        grid.Add(self.pos_text, pos=(3, 0), span=(1, 2), flag=wx.ALIGN_LEFT)
        
        self.SetSizer(grid)
        self.Fit()
        self.Show()

        self.pos_update()

    def pos_update(self):
        self.player.SetLabel(f'{self.board.pos.P.bit_count():02}')
        self.opponent.SetLabel(f'{self.board.pos.O.bit_count():02}')
        self.empty_count.SetLabel(f'{self.board.pos.empty_count():02}')
        self.moves.SetLabel(f'{possible_moves(self.board.pos).bit_count():02}')
        self.pos_text.SetValue(str(self.board.pos))

    def text_enter(self, event):
        self.board.pos = Position.from_string(self.pos_text.GetValue())

    def on_button_start(self, event):
        self.board.pos = Position.start()

    def on_button_clear(self, event):
        self.board.pos = Position()

    def on_button_flip(self, event):
        self.board.pos = Position(self.board.pos.O, self.board.pos.P)

    def on_button_eval(self, event):
        self.score.clear()
        solver = edax.Solver(Path(r'C:\Users\Dominic\Desktop\edax\4.3.2\bin\edax-4.4.exe'))
        for level in range(26):
            output = solver.solve(self.board.pos, level)
            depth = output[0].depth
            score = output[0].score
            sigma = solver.accuracy(self.board.pos.empty_count(), depth)
            self.score.add('edax', depth, WhiskedValue(score + sigma, score, score - sigma))
            self.Refresh()
            self.Update()


if __name__ == '__main__':
    app = wx.App()
    pos_frame = PositionFrame()
    app.MainLoop()