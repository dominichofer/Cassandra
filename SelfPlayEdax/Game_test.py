import unittest
from Position import *
from Game import *

class GameTest(unittest.TestCase):
    def test_str(self):
        game = Game(Position.Start(), [19, 18])
        self.assertEqual(str(game), str(Position.Start()) + " C4 C3")

    def test_Play(self):
        game = Game()
        game.Play(19)
        game.Play(18)
        self.assertEqual(game, Game(Position.Start(), [19, 18]))

    def test_PositionAt(self):
        game = Game()
        game.Play(19)
        pos = game.Position()
        game.Play(18)
        self.assertEqual(game.PositionAt(59), pos)

    def test_PositionAtNone(self):
        self.assertEqual(Game().PositionAt(59), None)

        
class ChildrenTest(unittest.TestCase):
    def test_Children_of_start(self):
        start = Game()
        self.assertEqual(sum(1 for _ in Children(start, 0)), 1)
        self.assertEqual(sum(1 for _ in Children(start, 1)), 4)
        self.assertEqual(sum(1 for _ in Children(start, 2)), 12)
        self.assertEqual(sum(1 for _ in Children(start, 3)), 56)
        self.assertEqual(sum(1 for _ in Children(start, 4)), 244)
        self.assertEqual(sum(1 for _ in Children(start, 5)), 1396)
        self.assertEqual(sum(1 for _ in Children(start, 6)), 8200)

if __name__ == '__main__':
    unittest.main()
