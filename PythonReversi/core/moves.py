from numpy import uint64
from itertools import islice


def first_set_field(b: uint64) -> int:
    return (int(b) & -int(b)).bit_length() - 1


def first_set_cleared(b: uint64) -> uint64:
    return b & (b - uint64(1))


class Iterator:
    def __init__(self, b: uint64):
        self.b = b

    def __next__(self):
        if self.b == 0:
            raise StopIteration
        move = first_set_field(self.b)
        self.b = first_set_cleared(self.b)
        return move


class Moves(uint64):
    def __new__(cls, b):
        return super(Moves, cls).__new__(cls, b)

    def __iter__(self):
        return Iterator(self)

    def __getitem__(self, index) -> int:
        return next(islice(self, index, None))

    def size(self) -> int:
        return self.bit_count()
