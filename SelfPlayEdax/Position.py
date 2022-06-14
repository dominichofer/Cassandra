import numpy as np
from typing import List
from copy import deepcopy
from collections.abc import Iterable


def popcount(b: np.uint64) -> int:
    return b.bit_count()

def FirstSetField(b: np.uint64) -> int:
    return (int(b) & -int(b)).bit_length() - 1

def FirstSetCleared(b: np.uint64) -> np.uint64:
    return b & (b - np.uint64(1))


def FieldToString(field: int) -> str:
    if field == 64:
        return 'PS'
    x = int(field / 8)
    y = field % 8
    return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][x] + str(y + 1)

def StringToField(string: str) -> int:
    if string.upper() == 'PS':
        return 64
    x = int(string[1]) - 1
    y = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'].index(string[0].upper())
    return y * 8 + x


class Moves:
    def __init__(self, b: np.uint64):
        self.__b = b

    def __eq__(self, o):
        return self.__b == o.__b
    def __neq__(self, o):
        return not self == o

    def __iter__(self):
        return self
    def __next__(self):
        if self.__b == 0:
            raise StopIteration
        move = FirstSetField(self.__b)
        self.__b = FirstSetCleared(self.__b)
        return move


class Position:
    def __init__(self, P = 0, O = 0):
        if not isinstance(P, np.uint64):
            P = np.uint64(P)
        if not isinstance(O, np.uint64):
            O = np.uint64(O)
        self.P = P
        self.O = O

    @staticmethod
    def Start():
        return Position(0x0000000810000000, 0x0000001008000000)

    def __eq__(self, o):
        return self.P == o.P and self.O == o.O
    def __neq__(self, o):
        return not self == o
    def __lt__(self, o):
        return self.P < o.P or (self.P == o.P and self.O < o.O)

    def __hash__(self):
        return hash((self.P, self.O))

    def __str__(self) -> str:
        def token(index):
            mask = np.uint64(1) << np.uint64(index)
            if self.P & mask:
                return 'X'
            elif self.O & mask:
                return 'O'
            else:
                return '-'

        return ''.join(token(i) for i in range(63, -1, -1))

    def Empties(self) -> np.uint64:
        return ~(self.P | self.O)

    def EmptyCount(self) -> int:
        return popcount(self.Empties())

    def Play(self, move: int):
        next = Play(self, move)
        self.P, self.O = next.P, next.O
        return self

    def PlayPass(self):
        self.P, self.O = self.O, self.P
        return self
    

def FlipCodiagonal(b):
    """
    # # # # # # # /
    # # # # # # / #
    # # # # # / # #
    # # # # / # # #
    # # # / # # # #
    # # / # # # # #
    # / # # # # # #
    / # # # # # # #<-LSB
    """
    if isinstance(b, Position):
        return Position(FlipCodiagonal(b.P), FlipCodiagonal(b.O))
    t  =  b ^ (b << np.uint64(36))
    b ^= (t ^ (b >> np.uint64(36))) & np.uint64(0xF0F0F0F00F0F0F0F)
    t  = (b ^ (b << np.uint64(18))) & np.uint64(0xCCCC0000CCCC0000)
    b ^=  t ^ (t >> np.uint64(18))
    t  = (b ^ (b << np.uint64( 9))) & np.uint64(0xAA00AA00AA00AA00)
    b ^=  t ^ (t >> np.uint64( 9))
    return b
        

def FlipDiagonal(b):
    """
    \ # # # # # # #
    # \ # # # # # #
    # # \ # # # # #
    # # # \ # # # #
    # # # # \ # # #
    # # # # # \ # #
    # # # # # # \ #
    # # # # # # # \<-LSB
    """
    if isinstance(b, Position):
        return Position(FlipDiagonal(b.P), FlipDiagonal(b.O))
    t  = (b ^ (b >> np.uint64( 7))) & np.uint64(0x00AA00AA00AA00AA)
    b ^=  t ^ (t << np.uint64( 7))
    t  = (b ^ (b >> np.uint64(14))) & np.uint64(0x0000CCCC0000CCCC)
    b ^=  t ^ (t << np.uint64(14))
    t  = (b ^ (b >> np.uint64(28))) & np.uint64(0x00000000F0F0F0F0)
    b ^=  t ^ (t << np.uint64(28))
    return b
        

def FlipHorizontal(b):
    """
    # # # #|# # # #
    # # # #|# # # #
    # # # #|# # # #
    # # # #|# # # #
    # # # #|# # # #
    # # # #|# # # #
    # # # #|# # # #
    # # # #|# # # #<-LSB
    """
    if isinstance(b, Position):
        return Position(FlipHorizontal(b.P), FlipHorizontal(b.O))
    b = ((b >> np.uint64(1)) & np.uint64(0x5555555555555555)) \
      | ((b << np.uint64(1)) & np.uint64(0xAAAAAAAAAAAAAAAA))
    b = ((b >> np.uint64(2)) & np.uint64(0x3333333333333333)) \
      | ((b << np.uint64(2)) & np.uint64(0xCCCCCCCCCCCCCCCC))
    b = ((b >> np.uint64(4)) & np.uint64(0x0F0F0F0F0F0F0F0F)) \
      | ((b << np.uint64(4)) & np.uint64(0xF0F0F0F0F0F0F0F0))
    return b


def FlipVertical(b):
    """
    # # # # # # # #
    # # # # # # # #
    # # # # # # # #
    # # # # # # # #
    ---------------
    # # # # # # # #
    # # # # # # # #
    # # # # # # # #
    # # # # # # # #<-LSB
    """
    if isinstance(b, Position):
        return Position(FlipVertical(b.P), FlipVertical(b.O))
    b = ((b >> np.uint64(32)) & np.uint64(0x00000000FFFFFFFF)) \
      | ((b << np.uint64(32)) & np.uint64(0xFFFFFFFF00000000))
    b = ((b >> np.uint64(16)) & np.uint64(0x0000FFFF0000FFFF)) \
      | ((b << np.uint64(16)) & np.uint64(0xFFFF0000FFFF0000))
    b = ((b >> np.uint64( 8)) & np.uint64(0x00FF00FF00FF00FF)) \
      | ((b << np.uint64( 8)) & np.uint64(0xFF00FF00FF00FF00))
    return b


def FlipToUnique(pos: Position):
    min = pos
    pos = FlipVertical(pos)
    if pos < min:
        min = pos
    pos = FlipHorizontal(pos)
    if pos < min:
        min = pos
    pos = FlipVertical(pos)
    if pos < min:
        min = pos
    pos = FlipCodiagonal(pos)
    if pos < min:
        min = pos
    pos = FlipVertical(pos)
    if pos < min:
        min = pos
    pos = FlipHorizontal(pos)
    if pos < min:
        min = pos
    pos = FlipVertical(pos)
    if pos < min:
        min = pos
    return min


def Play(pos: Position, move: int) -> Position:
    flips = Flips(pos, move)
    return Position(pos.O ^ flips, pos.P ^ flips ^ (np.uint64(1) << np.uint64(move)))


def PlayPass(pos:Position) -> Position:
    return Position(pos.O, pos.P)


def FlipsInOneDirection(pos:Position, x, y, dx, dy) -> np.uint64:
    flips = np.uint64(0)
    x += dx
    y += dy
    while (x >= 0) and (x < 8) and (y >= 0) and (y < 8):
        index = np.uint64(x * 8 + y)
        mask = np.uint64(1) << index
        if pos.O & mask:
            flips |= mask
        elif pos.P & mask:
            return flips
        else:
            break
        x += dx
        y += dy
    return np.uint64(0)


def Flips(pos: Position, move: int) -> np.uint64:
    x, y = int(move / 8), move % 8
    return FlipsInOneDirection(pos, x, y, -1, -1) \
         | FlipsInOneDirection(pos, x, y, -1, +0) \
         | FlipsInOneDirection(pos, x, y, -1, +1) \
         | FlipsInOneDirection(pos, x, y, +0, -1) \
         | FlipsInOneDirection(pos, x, y, +0, +1) \
         | FlipsInOneDirection(pos, x, y, +1, -1) \
         | FlipsInOneDirection(pos, x, y, +1, +0) \
         | FlipsInOneDirection(pos, x, y, +1, +1)


def PossibleMoves(pos:Position) -> Moves:
    maskO = pos.O & np.uint64(0x7E7E7E7E7E7E7E7E)
    
    flip1 = maskO & (pos.P << np.uint64(1))
    flip2 = maskO & (pos.P >> np.uint64(1))
    flip3 = pos.O & (pos.P << np.uint64(8))
    flip4 = pos.O & (pos.P >> np.uint64(8))
    flip5 = maskO & (pos.P << np.uint64(7))
    flip6 = maskO & (pos.P >> np.uint64(7))
    flip7 = maskO & (pos.P << np.uint64(9))
    flip8 = maskO & (pos.P >> np.uint64(9))

    flip1 |= maskO & (flip1 << np.uint64(1))
    flip2 |= maskO & (flip2 >> np.uint64(1))
    flip3 |= pos.O & (flip3 << np.uint64(8))
    flip4 |= pos.O & (flip4 >> np.uint64(8))
    flip5 |= maskO & (flip5 << np.uint64(7))
    flip6 |= maskO & (flip6 >> np.uint64(7))
    flip7 |= maskO & (flip7 << np.uint64(9))
    flip8 |= maskO & (flip8 >> np.uint64(9))

    mask1 = maskO & (maskO << np.uint64(1))
    mask2 =          mask1 >> np.uint64(1)
    mask3 = pos.O & (pos.O << np.uint64(8))
    mask4 =          mask3 >> np.uint64(8)
    mask5 = maskO & (maskO << np.uint64(7))
    mask6 =          mask5 >> np.uint64(7)
    mask7 = maskO & (maskO << np.uint64(9))
    mask8 =          mask7 >> np.uint64(9)

    flip1 |= mask1 & (flip1 << np.uint64( 2))
    flip2 |= mask2 & (flip2 >> np.uint64( 2))
    flip3 |= mask3 & (flip3 << np.uint64(16))
    flip4 |= mask4 & (flip4 >> np.uint64(16))
    flip5 |= mask5 & (flip5 << np.uint64(14))
    flip6 |= mask6 & (flip6 >> np.uint64(14))
    flip7 |= mask7 & (flip7 << np.uint64(18))
    flip8 |= mask8 & (flip8 >> np.uint64(18))

    flip1 |= mask1 & (flip1 << np.uint64( 2))
    flip2 |= mask2 & (flip2 >> np.uint64( 2))
    flip3 |= mask3 & (flip3 << np.uint64(16))
    flip4 |= mask4 & (flip4 >> np.uint64(16))
    flip5 |= mask5 & (flip5 << np.uint64(14))
    flip6 |= mask6 & (flip6 >> np.uint64(14))
    flip7 |= mask7 & (flip7 << np.uint64(18))
    flip8 |= mask8 & (flip8 >> np.uint64(18))

    flip1 <<= np.uint64(1)
    flip2 >>= np.uint64(1)
    flip3 <<= np.uint64(8)
    flip4 >>= np.uint64(8)
    flip5 <<= np.uint64(7)
    flip6 >>= np.uint64(7)
    flip7 <<= np.uint64(9)
    flip8 >>= np.uint64(9)

    return Moves(pos.Empties() & (flip1 | flip2 | flip3 | flip4 | flip5 | flip6 | flip7 | flip8))


def OpponentPossibleMoves(pos:Position) -> Moves:
    return PossibleMoves(PlayPass(pos))


def IsPass(pos: Position) -> bool:
    return not PossibleMoves(pos) and OpponentPossibleMoves(pos)


def IsGameOver(pos: Position) -> bool:
    return not PossibleMoves(pos) and not OpponentPossibleMoves(pos)


def Children(pos: Position, empty_count_diff: int = 1):
    if empty_count_diff == 0:
        yield pos
        return
    if IsGameOver(pos):
        return
    if IsPass(pos):
        pos.PlayPass()
    for move in PossibleMoves(pos):
        yield from Children(Play(pos, move), empty_count_diff - 1)


def AllUniqueChildren(pos: Position, empty_count_diff: int = 1):
    if empty_count_diff == 0:
        return { pos }
    return {
        FlipToUnique(p)
        for prev in AllUniqueChildren(pos, empty_count_diff - 1)
        for p in Children(prev)
        }


class PositionScore:    
    def __init__(self, pos: Position = Position.Start(), score: int = +70):
        self.pos = pos
        self.score = score

    def __eq__(self, o):
        if isinstance(o, Position):
            return self.pos == o and self.score == +70
        return self.pos == o.pos and self.score == o.score

    def __neq__(self, o):
        return not self == o

    def __str__(self) -> str:
        if self.HasScore():
            return f'{self.pos} X % {self.score:+03}'
        else:
            return f'{self.pos} X'

    def HasScore(self) -> bool:
        return self.score != +70