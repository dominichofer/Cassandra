import unittest
import Position

class BitBoard(unittest.TestCase):

    def test_FromField_one_arg(self):
        self.assertEqual(Position.BitBoard.FromField(1), Position.BitBoard(2))

    def test_FromField_two_arg(self):
        self.assertEqual(Position.BitBoard.FromField(1,0), Position.BitBoard(2))

    def test_FromField_tuple(self):
        self.assertEqual(Position.BitBoard.FromField((1,0)), Position.BitBoard(2))

if __name__ == '__main__':
    unittest.main()
