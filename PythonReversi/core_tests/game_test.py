import unittest
from core import Game, Position, is_game

class GameTest(unittest.TestCase):
    def test_str(self):
        game = Game(Position.start(), [19, 18])
        self.assertEqual(game, Game.from_string(str(game)))

    def test_play(self):
        game = Game(Position.start())
        game.play(19)
        game.play(18)
        self.assertEqual(game, Game(Position.start(), [19, 18]))

    def test_positions(self):
        game = Game(Position.start())
        pos1 = game.current_position
        game.play(19)
        pos2 = game.current_position
        game.play(18)
        pos3 = game.current_position

        self.assertEqual([p for p in game.positions()], [pos1, pos2, pos3])
        

class IsGame(unittest.TestCase):
    def test_zero_ply(self):
        self.assertTrue(is_game('---------------------------OOO-----OOO-----OXO------X----------- X'))

    def test_one_ply(self):
        self.assertTrue(is_game('---------------------------OOO-----OOO-----OXO------X----------- X G6'))

    def test_two_ply(self):
        self.assertTrue(is_game('---------------------------OOO-----OOO-----OXO------X----------- X G6 D7'))


        
#class ChildrenTest(unittest.TestCase):
#    def test_Children_of_start(self):
#        start = Game()
#        self.assertEqual(sum(1 for _ in Children(start, 0)), 1)
#        self.assertEqual(sum(1 for _ in Children(start, 1)), 4)
#        self.assertEqual(sum(1 for _ in Children(start, 2)), 12)
#        self.assertEqual(sum(1 for _ in Children(start, 3)), 56)
#        self.assertEqual(sum(1 for _ in Children(start, 4)), 244)
#        self.assertEqual(sum(1 for _ in Children(start, 5)), 1396)
#        self.assertEqual(sum(1 for _ in Children(start, 6)), 8200)


if __name__ == '__main__':
    unittest.main(verbosity=2)
