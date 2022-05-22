class BitBoard:
    def __init__(self, b:int=0):
        self.__b = b
        
    @staticmethod
    def FromField(*index):
        if len(index) == 1:
            index = index[0]
        if isinstance(index, tuple):
            x, y = index
            index = x + 8 * y
        return BitBoard(1 << index)

    def __eq__(self, o):
        return self.__b == o.__b
    def __neq__(self, o):
        return not self == o

    def __and__(self, o):
        return BitBoard(self.__b & o.__b)
    def __or__(self, o):
        return BitBoard(self.__b | o.__b)
    def __xor__(self, o):
        return BitBoard(self.__b ^ o.__b)
    def __invert__(self):
        return BitBoard(~self.__b)

    def __lshift__(self, i):
        return BitBoard(self.__b << i)
    def __rshift__(self, i):
        return BitBoard(self.__b >> i)

    def __getitem__(self, index) -> bool:
        return bool(self.__b & BitBoard.FromField(index).__b)
    def __setitem__(self, index, value:bool):
        if value:
            self.__b |= BitBoard.FromField(index).__b
        else:
            self.__b &= ~BitBoard.FromField(index).__b

class Position:
    def __init__(self, P = BitBoard(), O = BitBoard()):
        if not isinstance(P, BitBoard):
            P = BitBoard(P)
        if not isinstance(O, BitBoard):
            O = BitBoard(O)
        self.P = P
        self.O = O

    def __eq__(self, o):
        return self.P == o.P and self.O == o.O
    def __neq__(self, o):
        return not self == o

    @staticmethod
    def Start():
        return Position(0x0000000810000000, 0x0000001008000000)

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
    
    flip1 = maskO & (pos.P << 1)
    flip2 = maskO & (pos.P >> 1)
    flip3 = pos.O & (pos.P << 8)
    flip4 = pos.O & (pos.P >> 8)
    flip5 = maskO & (pos.P << 7)
    flip6 = maskO & (pos.P >> 7)
    flip7 = maskO & (pos.P << 9)
    flip8 = maskO & (pos.P >> 9)

    flip1 |= maskO & (flip1 << 1)
    flip2 |= maskO & (flip2 >> 1)
    flip3 |= pos.O & (flip3 << 8)
    flip4 |= pos.O & (flip4 >> 8)
    flip5 |= maskO & (flip5 << 7)
    flip6 |= maskO & (flip6 >> 7)
    flip7 |= maskO & (flip7 << 9)
    flip8 |= maskO & (flip8 >> 9)

    mask1 = maskO & (maskO << 1)
    mask2 = mask1 >> 1
    mask3 = pos.O & (pos.O << 8)
    mask4 = mask3 >> 8
    mask5 = maskO & (maskO << 7)
    mask6 = mask5 >> 7
    mask7 = maskO & (maskO << 9)
    mask8 = mask7 >> 9

    flip1 |= mask1 & (flip1 << 2)
    flip2 |= mask2 & (flip2 >> 2)
    flip3 |= mask3 & (flip3 << 16)
    flip4 |= mask4 & (flip4 >> 16)
    flip5 |= mask5 & (flip5 << 14)
    flip6 |= mask6 & (flip6 >> 14)
    flip7 |= mask7 & (flip7 << 18)
    flip8 |= mask8 & (flip8 >> 18)

    flip1 |= mask1 & (flip1 << 2)
    flip2 |= mask2 & (flip2 >> 2)
    flip3 |= mask3 & (flip3 << 16)
    flip4 |= mask4 & (flip4 >> 16)
    flip5 |= mask5 & (flip5 << 14)
    flip6 |= mask6 & (flip6 >> 14)
    flip7 |= mask7 & (flip7 << 18)
    flip8 |= mask8 & (flip8 >> 18)

    flip1 <<= 1
    flip2 >>= 1
    flip3 <<= 8
    flip4 >>= 8
    flip5 <<= 7
    flip6 >>= 7
    flip7 <<= 9
    flip8 >>= 9

    return ~(pos.P | pos.O) & (flip1 | flip2 | flip3 | flip4 | flip5 | flip6 | flip7 | flip8)