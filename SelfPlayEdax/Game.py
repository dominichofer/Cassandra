from copy import deepcopy
from collections.abc import Iterable
from pathlib import Path
from Position import *

class Game:
    def __init__(self, start: Position = Position.Start(), moves = None):
        self.__start = deepcopy(start)
        self.__current = deepcopy(start)
        self.__moves = []
        for move in (moves or []):
            self.Play(move)
                
    def __eq__(self, o):
        return self.__start == o.__start and self.__moves == o.__moves
    def __neq__(self, o):
        return not self == o

    def __str__(self) -> str:
        return str(self.__start) + ' ' + ' '.join(FieldToString(m) for m in self.__moves)

    def StartPosition(self) -> Position:
        return deepcopy(self.__start)

    def Position(self) -> Position:
        return deepcopy(self.__current)

    def Moves(self) -> list:
        return deepcopy(self.__moves)

    def Play(self, move):
        self.__moves.append(move)
        self.__current.Play(move)
        if IsPass(self.__current):
            self.__current.PlayPass()

    def EmptyCount(self) -> int:
        return self.__current.EmptyCount()

    def PossibleMoves(self) -> Moves:
        return PossibleMoves(self.__current)

    def PositionAt(self, empty_count: int):
        pos = deepcopy(self.__start)
        if pos.EmptyCount() == empty_count:
            return pos
        for move in self.__moves:
            pos.Play(move)
            if pos.EmptyCount() == empty_count:
                return pos
        return None

    def Positions(self):
        pos = deepcopy(self.__start)
        yield pos
        for move in self.__moves:
            pos.Play(move)
            yield pos


    def IsOver(self) -> bool:
        return IsGameOver(self.__current)


def Play(game: Game, move) -> Game:
    ret = deepcopy(game)
    ret.Play(move)
    return ret


def Children(game: Game, empty_count_diff: int = 1):
    if empty_count_diff == 0:
        yield game
        return
    if game.IsOver():
        return
    for move in game.PossibleMoves():
        yield from Children(Play(game, move), empty_count_diff - 1)
        

def WriteToFile(data, file_path: Path):
    if isinstance(data, Iterable):
        file_path.write_text('\n'.join(str(Game(d)) for d in data))
    else:
        file_path.write_text(str(Game(data)))
