import numpy

class BitBoard:
    def __init__(self, b:numpy.uint64=0):
        self.__b = b

    def __eq__(self, o):
        return self.__b == o.__b
    def __neq__(self, o):
        return self.__b != o.__b

    def __and__(self, o):
        return BitBoard(self.__b & o.__b)
    def __or__(self, o):
        return BitBoard(self.__b | o.__b)
    def __invert__(self):
        return BitBoard(~self.__b)

    def __lshift__(self, i):
        return BitBoard(self.__b << i)
    def __rshift__(self, i):
        return BitBoard(self.__b >> i)

    def __getitem__(self, key) -> bool:
        x,y = key
        return bool(self.__b & (1 << (63 - x - 8 * y)))
    def __setitem__(self, key, value:bool):
        x,y = key
        if value:
            self.__b |= (1 << (63 - x - 8 * y))
        else:
            self.__b &= ~(1 << (63 - x - 8 * y))

class Position:
    def __init__(self, P=BitBoard(), O=BitBoard()):
        if not isinstance(P, BitBoard):
            P = BitBoard(P)
        if not isinstance(O, BitBoard):
            O = BitBoard(O)
        self.P = P
        self.O = O
