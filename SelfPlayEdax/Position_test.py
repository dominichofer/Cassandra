import unittest
import numpy as np
from copy import deepcopy
from Position import *


class BitBoardTest(unittest.TestCase):
    def test_popcount(self):
        self.assertEqual(popcount(0), 0)
        self.assertEqual(popcount(1), 1)
        self.assertEqual(popcount(~np.uint64(0)), 64)

    def test_FlipCodiagonal(self):
        self.assertEqual(FlipCodiagonal(0x8080808000000000), 0x000000000000000F)

    def test_FlipDiagonal(self):
        self.assertEqual(FlipDiagonal(0x8080808000000000), 0xF000000000000000)

    def test_FlipHorizontal(self):
        self.assertEqual(FlipHorizontal(0x8080808000000000), 0x0101010100000000)

    def test_FlipVertical(self):
        self.assertEqual(FlipVertical(0x8080808000000000), 0x0000000080808080)


class FieldTest(unittest.TestCase):
    def test_string_roundtrip(self):
        for i in range(65):
            self.assertEqual(StringToField(FieldToString(i)), i)


class PositionTest(unittest.TestCase):
    def test_Start(self):
        start = Position.Start()
        self.assertEqual(FlipCodiagonal(start), start)
        self.assertEqual(FlipDiagonal(start), start)
        self.assertEqual(FlipVertical(FlipHorizontal(start)), start)
        self.assertEqual(FlipHorizontal(FlipVertical(start)), start)
        self.assertEqual(start.EmptyCount(), 60)

    def test_string(self):
        self.assertEqual(str(Position.Start()), '---------------------------OX------XO---------------------------')

    def test_Empties(self):
        self.assertEqual(Position().Empties(), 0xFFFFFFFFFFFFFFFF)

    def test_EmptyCount(self):
        self.assertEqual(Position().EmptyCount(), 64)

    def test_FlipCodiagonal(self):
        self.assertEqual(FlipCodiagonal(Position(0x8080808000000000, 0x4040404000000000)),
                        Position(0x000000000000000F, 0x0000000000000F00))

    def test_FlipDiagonal(self):
        self.assertEqual(FlipDiagonal(Position(0x8080808000000000, 0x4040404000000000)),
                        Position(0xF000000000000000, 0x00F0000000000000))

    def test_FlipHorizontal(self):
        self.assertEqual(FlipHorizontal(Position(0x8080808000000000, 0x4040404000000000)),
                        Position(0x0101010100000000, 0x0202020200000000))

    def test_FlipVertical(self):
        self.assertEqual(FlipVertical(Position(0x8080808000000000, 0x4040404000000000)),
                        Position(0x0000000080808080, 0x0000000040404040))

    def test_FlipToUnique(self):
        pos1 = Position(0x8080808000000000, 0x4040404000000000)
        pos2 = FlipVertical(pos1)

        pos1 = FlipToUnique(pos1)
        pos2 = FlipToUnique(pos2)

        self.assertEqual(pos1, pos2)


class PlayTest(unittest.TestCase):
    def test_PlayPass(self):
        start = Position.Start()
        self.assertEqual(PlayPass(start), FlipHorizontal(start))

    def test_Play(self):
        start = Position.Start()
        pm = PossibleMoves(start)
        move = next(pm)
        pos = Play(start, move)
        self.assertEqual(pos, Position(0x0000001000000000, 0x0000000818080000))

    def test_PossibleMoves(self):
        self.assertEqual(PossibleMoves(Position.Start()), Moves(0x0000102004080000))


class ChildrenTest(unittest.TestCase):
    def test_Children_of_start(self):
        start = Position.Start()
        self.assertEqual(sum(1 for _ in Children(start, 0)), 1)
        self.assertEqual(sum(1 for _ in Children(start, 1)), 4)
        self.assertEqual(sum(1 for _ in Children(start, 2)), 12)
        self.assertEqual(sum(1 for _ in Children(start, 3)), 56)
        self.assertEqual(sum(1 for _ in Children(start, 4)), 244)
        self.assertEqual(sum(1 for _ in Children(start, 5)), 1396)
        self.assertEqual(sum(1 for _ in Children(start, 6)), 8200)


class AllUniqueChildrenTest(unittest.TestCase):
    def test_AllUniqueChildren_of_start(self):
        start = Position.Start()
        self.assertEqual(len(AllUniqueChildren(start, 0)), 1)
        self.assertEqual(len(AllUniqueChildren(start, 1)), 1)
        self.assertEqual(len(AllUniqueChildren(start, 2)), 3)
        self.assertEqual(len(AllUniqueChildren(start, 3)), 14)
        self.assertEqual(len(AllUniqueChildren(start, 4)), 60)
        self.assertEqual(len(AllUniqueChildren(start, 5)), 322)
        self.assertEqual(len(AllUniqueChildren(start, 6)), 1773)


if __name__ == '__main__':
    unittest.main()
