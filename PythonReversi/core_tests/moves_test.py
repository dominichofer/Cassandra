import unittest
from core import Moves


class MovesTest(unittest.TestCase):
    def test_iter(self):
        moves = Moves(0x8000000000000001)
        self.assertEqual([m for m in moves], [0, 63])

    def test_getitem(self):
        moves = Moves(0x8000000000000001)
        self.assertEqual(moves[0], 0)
        self.assertEqual(moves[1], 63)

    def test_bool(self):
        self.assertFalse(Moves(0x0000000000000000))
        self.assertTrue(Moves(0x8000000000000001))

    def test_size(self):
        self.assertEqual(Moves(0x0000000000000000).size(), 0)
        self.assertEqual(Moves(0x8000000000000001).size(), 2)
        self.assertEqual(Moves(0xFFFFFFFFFFFFFFFF).size(), 64)


if __name__ == '__main__':
    unittest.main(verbosity=2)
