from typing import List
from collections.abc import Iterable
from pathlib import Path
from Position import Position, PositionScore
import numpy as np


def ParseLine(string: str) -> PositionScore:
    board, player, *rest = string.split(' ')

    X = np.uint64(0)
    O = np.uint64(0)
    for c in board:
        X <<= np.uint64(1)
        O <<= np.uint64(1)
        if c == 'X':
            X += np.uint64(1)
        if c == 'O':
            O += np.uint64(1)
     
    ps = PositionScore()
    if player == 'X':
        ps.pos = Position(X, O)
    else:
        ps.pos = Position(O, X)
    if len(rest) == 2:
        ps.score = int(rest[1])
    return ps


def ParseFile(file_path: Path) -> List[PositionScore]:
    return [ParseLine(line) for line in file_path.read_text().strip().split('\n')]


def WriteToFile(data, file_path: Path):
    if isinstance(data, Iterable):
        file_path.write_text('\n'.join(str(PositionScore(d)) for d in data))
    else:
        file_path.write_text(str(PositionScore(data)))
