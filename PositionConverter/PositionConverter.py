import PySimpleGUI as sg
from Position import *

#board_layout = [[sg.T('     ')] + [sg.T('{}'.format(a), pad=((23,27),0), font='Any 13') for a in 'abcdefgh']]
## loop though board and create buttons with images
#for i in range(8):
#    row = [sg.T(str(8-i)+'   ', font='Any 13')]
#    for j in range(8):
#        piece_image = images[board[i][j]]
#        row.append(render_square(piece_image, key=(i,j), location=(i,j)))
#    row.append(sg.T(str(8-i)+'   ', font='Any 13'))
#    board_layout.append(row)
## add the labels across bottom of board
#board_layout.append([sg.T('     ')] + [sg.T('{}'.format(a), pad=((23,27),0), font='Any 13') for a in 'abcdefgh'])

class Model:
    self.pos = Position()

board = [[sg.RButton(' ', image_filename='empty.png', border_width=0, pad=(0, 0), key=(col,row)) for col in range(8)] for row in range(8)]

layout = [board]
window = sg.Window('Reversi - Position', layout)

while (True):
    event, values = window.read()
    button = window.find_element(key=event)
    button.Update(image_filename='white.png')
    window.Refresh()