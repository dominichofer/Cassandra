import unittest
from core import Position, PositionScore, is_position_score


class PositionScoreTest(unittest.TestCase):
    def test_from_string_score(self):
        self.assertEqual(
            PositionScore.from_string('O--------------------------------------------------------------X O % +02'),
            PositionScore(Position(1 << 63, 1), +2)
           )

    def test_from_string_no_score(self):
        self.assertEqual(
            PositionScore.from_string('O--------------------------------------------------------------X O'),
            PositionScore(Position(1 << 63, 1))
        )
        
    def test_str(self):
        ps = PositionScore(Position.start(), -7)
        self.assertEqual(str(ps), '---------------------------OX------XO--------------------------- X % -07')


class IsPositionScore(unittest.TestCase):
    def test_X_player(self):
        self.assertTrue(is_position_score('O--------------------------------------------------------------X X % +60'))

    def test_O_player(self):
        self.assertTrue(is_position_score('O--------------------------------------------------------------X O % -60'))

    def test_no_player(self):
        self.assertFalse(is_position_score('O--------------------------------------------------------------X % +60'))


if __name__ == '__main__':
    unittest.main(verbosity=2)