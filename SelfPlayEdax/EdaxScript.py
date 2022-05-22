from Position import Position, BitBoard
from dataclasses import dataclass
from typing import List
            
@dataclass
class PositionScore:
    pos: Position = Position()
    score: int = +70 #undefined score

    def HasScore(self) -> bool:
        return self.score != +70

def ParseMove(string: str):
    if string == 'ps':
        return None #pass
    return ['H','G','F','E','D','C','B','A'].index(string[0].upper()), 8 - int(string[1])

def ParseLine(string: str) -> PositionScore:
    board, player, *rest = string.split(' ')

    X = BitBoard()
    O = BitBoard()
    for i, c in enumerate(board):
        X[63 - i] = (c == 'X')
        O[63 - i] = (c == 'O')
     
    ps = PositionScore()
    if player == 'X':
        ps.pos = Position(X, O)
    else:
        ps.pos = Position(O, X)
    if len(rest) == 2:
        ps.score = int(rest[1])
    return ps

def ParseFile(path: str) -> List[PositionScore]:
    with open(path, 'r') as file:
        return [ParseLine(line) for line in file.read().strip().split('\n')]

def ToString(ps: PositionScore) -> str:
    if isinstance(ps, Position):
        ps = PositionScore(ps)
    pos = str()
    for i in range(64):
        if ps.pos.P[63 - i]:
            pos += 'X'
        elif ps.pos.O[63 - i]:
            pos += 'O'
        else:
            pos += '-'
    
    if ps.HasScore():
        return f'{pos} X % {ps.score:+03}\n'
    else:
        return f'{pos} X\n'

def WriteToFile(data, path: str):
    with open(path, 'w') as file:
        for d in data:
            file.write(ToString(d))

class File:
    def __init__(self, path: str):
        self.path: str = path
        self.lines: List[PositionScore] = ParseFile(path)

    def __getitem__(self, index) -> PositionScore:
        return self.lines[index]
    def __setitem__(self, index, value:PositionScore):
        self.lines[index] = value

    def WriteTo(self, path:str):
        WriteToFile(self.lines, path)

    def WriteBack(self):
        self.WriteTo(self.path)
