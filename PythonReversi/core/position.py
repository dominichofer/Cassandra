from turtle import pos
from numpy import uint64
from .moves import Moves
import re


class Position:

    def __init__(self, P = 0, O = 0):
        self.P: uint64 = uint64(P)
        self.O: uint64 = uint64(O)

    @staticmethod
    def start():
        return Position(0x0000000810000000, 0x0000001008000000)

    @staticmethod
    def from_string(s: str):
        # Example input:
        # 'OO-XXXX-OOOOOXX-OOOOXOXOOXOXOXXXOXOOOXXXOXOXOXXXOOXXXXXXOOOOOOOO X'
        # 'OO-XXXX-OOOOOXX-OOOOXOXOOXOXOXXXOXOOOXXXOXOXOXXXOOXXXXXXOOOOOOOO O'

        P = uint64(0)
        O = uint64(0)
        for i in range(64):
            if s[i] == 'X':
                P |= uint64(1) << uint64(63 - i)
            if s[i] == 'O':
                O |= uint64(1) << uint64(63 - i)

        if s[65] == 'O':
            return Position(O, P)
        else:
            return Position(P, O)

    def __str__(self) -> str:
        def token(i):
            mask = uint64(1) << uint64(63 - i)
            if self.P & mask:
                return 'X'
            elif self.O & mask:
                return 'O'
            else:
                return '-'

        return ''.join(token(i) for i in range(64)) + ' X'

    def __eq__(self, o):
        return self.P == o.P and self.O == o.O

    def __lt__(self, o):
        return self.P < o.P or (self.P == o.P and self.O < o.O)

    def __hash__(self):
        return hash((self.P, self.O))

    def empties(self) -> uint64:
        return ~(self.P | self.O)

    def empty_count(self) -> int:
        return self.empties().bit_count()

    
def is_position(s: str):
    return re.fullmatch(r'[XO-]{64} [XO]', s) is not None


#def flipped_codiagonal(b):
#    """
#    # # # # # # # /
#    # # # # # # / #
#    # # # # # / # #
#    # # # # / # # #
#    # # # / # # # #
#    # # / # # # # #
#    # / # # # # # #
#    / # # # # # # # <-LSB
#    """
#    if isinstance(b, Position):
#        return Position(flipped_codiagonal(b.P), flipped_codiagonal(b.O))
#    t  =  b ^ (b << uint64(36))
#    b ^= (t ^ (b >> uint64(36))) & uint64(0xF0F0F0F00F0F0F0F)
#    t  = (b ^ (b << uint64(18))) & uint64(0xCCCC0000CCCC0000)
#    b ^=  t ^ (t >> uint64(18))
#    t  = (b ^ (b << uint64( 9))) & uint64(0xAA00AA00AA00AA00)
#    b ^=  t ^ (t >> uint64( 9))
#    return b
        

#def flipped_diagonal(b):
#    """
#    \ # # # # # # #
#    # \ # # # # # #
#    # # \ # # # # #
#    # # # \ # # # #
#    # # # # \ # # #
#    # # # # # \ # #
#    # # # # # # \ #
#    # # # # # # # \ <-LSB
#    """
#    if isinstance(b, Position):
#        return Position(flipped_diagonal(b.P), flipped_diagonal(b.O))
#    t  = (b ^ (b >> uint64( 7))) & uint64(0x00AA00AA00AA00AA)
#    b ^=  t ^ (t << uint64( 7))
#    t  = (b ^ (b >> uint64(14))) & uint64(0x0000CCCC0000CCCC)
#    b ^=  t ^ (t << uint64(14))
#    t  = (b ^ (b >> uint64(28))) & uint64(0x00000000F0F0F0F0)
#    b ^=  t ^ (t << uint64(28))
#    return b
        

#def flipped_horizontal(b):
#    """
#    # # # #|# # # #
#    # # # #|# # # #
#    # # # #|# # # #
#    # # # #|# # # #
#    # # # #|# # # #
#    # # # #|# # # #
#    # # # #|# # # #
#    # # # #|# # # # <-LSB
#    """
#    if isinstance(b, Position):
#        return Position(flipped_horizontal(b.P), flipped_horizontal(b.O))
#    b = ((b >> uint64(1)) & uint64(0x5555555555555555)) \
#      | ((b << uint64(1)) & uint64(0xAAAAAAAAAAAAAAAA))
#    b = ((b >> uint64(2)) & uint64(0x3333333333333333)) \
#      | ((b << uint64(2)) & uint64(0xCCCCCCCCCCCCCCCC))
#    b = ((b >> uint64(4)) & uint64(0x0F0F0F0F0F0F0F0F)) \
#      | ((b << uint64(4)) & uint64(0xF0F0F0F0F0F0F0F0))
#    return b


#def flipped_vertical(b):
#    """
#    # # # # # # # #
#    # # # # # # # #
#    # # # # # # # #
#    # # # # # # # #
#    ---------------
#    # # # # # # # #
#    # # # # # # # #
#    # # # # # # # #
#    # # # # # # # # <-LSB
#    """
#    if isinstance(b, Position):
#        return Position(flipped_vertical(b.P), flipped_vertical(b.O))
#    b = ((b >> uint64(32)) & uint64(0x00000000FFFFFFFF)) \
#      | ((b << uint64(32)) & uint64(0xFFFFFFFF00000000))
#    b = ((b >> uint64(16)) & uint64(0x0000FFFF0000FFFF)) \
#      | ((b << uint64(16)) & uint64(0xFFFF0000FFFF0000))
#    b = ((b >> uint64( 8)) & uint64(0x00FF00FF00FF00FF)) \
#      | ((b << uint64( 8)) & uint64(0xFF00FF00FF00FF00))
#    return b


#def flipped_to_unique(pos: Position):
#    min = pos
#    pos = flipped_vertical(pos)
#    if pos < min:
#        min = pos
#    pos = flipped_horizontal(pos)
#    if pos < min:
#        min = pos
#    pos = flipped_vertical(pos)
#    if pos < min:
#        min = pos
#    pos = flipped_codiagonal(pos)
#    if pos < min:
#        min = pos
#    pos = flipped_vertical(pos)
#    if pos < min:
#        min = pos
#    pos = flipped_horizontal(pos)
#    if pos < min:
#        min = pos
#    pos = flipped_vertical(pos)
#    if pos < min:
#        min = pos
#    return min


def flips_in_one_direction(pos:Position, x, y, dx, dy) -> uint64:
    bits = uint64(0)
    x += dx
    y += dy
    while (x >= 0) and (x < 8) and (y >= 0) and (y < 8):
        index = uint64(x * 8 + y)
        mask = uint64(1) << index
        if pos.O & mask:
            bits |= mask
        elif pos.P & mask:
            return bits
        else:
            break
        x += dx
        y += dy
    return uint64(0)


def flips(pos: Position, move: int) -> uint64:
    x, y = int(move / 8), move % 8
    return flips_in_one_direction(pos, x, y, -1, -1) \
         | flips_in_one_direction(pos, x, y, -1, +0) \
         | flips_in_one_direction(pos, x, y, -1, +1) \
         | flips_in_one_direction(pos, x, y, +0, -1) \
         | flips_in_one_direction(pos, x, y, +0, +1) \
         | flips_in_one_direction(pos, x, y, +1, -1) \
         | flips_in_one_direction(pos, x, y, +1, +0) \
         | flips_in_one_direction(pos, x, y, +1, +1)


def play(pos: Position, move: int) -> Position:
    bits = flips(pos, move)
    return Position(pos.O ^ bits, pos.P ^ bits ^ (uint64(1) << uint64(move)))


def play_pass(pos: Position) -> Position:
    return Position(pos.O, pos.P)


def play_or_pass(pos: Position, move: int) -> Position:
    if move == 64:
        return play_pass(pos)
    return play(pos, move)


def auto_pass(pos: Position) -> Position:
    if not possible_moves(pos):
        passed = play_pass(pos)
        if possible_moves(passed):
            return passed
    return pos


def possible_moves(pos: Position) -> Moves:
    maskO = pos.O & uint64(0x7E7E7E7E7E7E7E7E)
    
    flip1 = maskO & (pos.P << uint64(1))
    flip2 = maskO & (pos.P >> uint64(1))
    flip3 = pos.O & (pos.P << uint64(8))
    flip4 = pos.O & (pos.P >> uint64(8))
    flip5 = maskO & (pos.P << uint64(7))
    flip6 = maskO & (pos.P >> uint64(7))
    flip7 = maskO & (pos.P << uint64(9))
    flip8 = maskO & (pos.P >> uint64(9))

    flip1 |= maskO & (flip1 << uint64(1))
    flip2 |= maskO & (flip2 >> uint64(1))
    flip3 |= pos.O & (flip3 << uint64(8))
    flip4 |= pos.O & (flip4 >> uint64(8))
    flip5 |= maskO & (flip5 << uint64(7))
    flip6 |= maskO & (flip6 >> uint64(7))
    flip7 |= maskO & (flip7 << uint64(9))
    flip8 |= maskO & (flip8 >> uint64(9))

    mask1 = maskO & (maskO << uint64(1))
    mask2 =          mask1 >> uint64(1)
    mask3 = pos.O & (pos.O << uint64(8))
    mask4 =          mask3 >> uint64(8)
    mask5 = maskO & (maskO << uint64(7))
    mask6 =          mask5 >> uint64(7)
    mask7 = maskO & (maskO << uint64(9))
    mask8 =          mask7 >> uint64(9)

    flip1 |= mask1 & (flip1 << uint64( 2))
    flip2 |= mask2 & (flip2 >> uint64( 2))
    flip3 |= mask3 & (flip3 << uint64(16))
    flip4 |= mask4 & (flip4 >> uint64(16))
    flip5 |= mask5 & (flip5 << uint64(14))
    flip6 |= mask6 & (flip6 >> uint64(14))
    flip7 |= mask7 & (flip7 << uint64(18))
    flip8 |= mask8 & (flip8 >> uint64(18))

    flip1 |= mask1 & (flip1 << uint64( 2))
    flip2 |= mask2 & (flip2 >> uint64( 2))
    flip3 |= mask3 & (flip3 << uint64(16))
    flip4 |= mask4 & (flip4 >> uint64(16))
    flip5 |= mask5 & (flip5 << uint64(14))
    flip6 |= mask6 & (flip6 >> uint64(14))
    flip7 |= mask7 & (flip7 << uint64(18))
    flip8 |= mask8 & (flip8 >> uint64(18))

    flip1 <<= uint64(1)
    flip2 >>= uint64(1)
    flip3 <<= uint64(8)
    flip4 >>= uint64(8)
    flip5 <<= uint64(7)
    flip6 >>= uint64(7)
    flip7 <<= uint64(9)
    flip8 >>= uint64(9)

    return Moves(pos.empties() & (flip1 | flip2 | flip3 | flip4 | flip5 | flip6 | flip7 | flip8))


#def opponent_possible_moves(pos:Position) -> Moves:
#    return possible_moves(play_pass(pos))


#def IsPass(pos: Position) -> bool:
#    return not possible_moves(pos) and opponent_possible_moves(pos)


#def IsGameOver(pos: Position) -> bool:
#    return not possible_moves(pos) and not opponent_possible_moves(pos)


#def Children(pos: Position, empty_count_diff: int = 1):
#    if empty_count_diff == 0:
#        yield pos
#        return
#    if IsGameOver(pos):
#        return
#    if IsPass(pos):
#        pos.play_pass()
#    for move in possible_moves(pos):
#        yield from Children(play(pos, move), empty_count_diff - 1)


#def AllUniqueChildren(pos: Position, empty_count_diff: int = 1):
#    if empty_count_diff == 0:
#        return {pos}
#    return {
#        flipped_to_unique(p)
#        for prev in AllUniqueChildren(pos, empty_count_diff - 1)
#        for p in Children(prev)
#    }
