import unittest
from core import GameScore, Position, undefined_score

class GameScoreTest(unittest.TestCase):
    def test_str(self):
        gs = GameScore(Position.start(), [19, 18], [+1, -1, +1])
        self.assertEqual(gs, GameScore.from_string(str(gs)))

    def test_play(self):
        gs = GameScore(Position.start())
        gs.play(19)
        gs.play(18)
        self.assertEqual(gs, GameScore(Position.start(), [19, 18], [undefined_score] * 3))


if __name__ == '__main__':
    unittest.main(verbosity=2)
