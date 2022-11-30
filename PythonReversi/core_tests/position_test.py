import unittest
from core import Position, play, play_pass, possible_moves, Moves, is_position, is_position_score


class PositionTest(unittest.TestCase):
    def test_start(self):
        self.assertEqual(Position.start().empty_count(), 60)

    def test_from_string_X_to_play(self):
        pos = Position.from_string('O--------------------------------------------------------------X X')
        self.assertEqual(pos, Position(1, 1 << 63))

    def test_from_string_O_to_play(self):
        pos = Position.from_string('O--------------------------------------------------------------X O')
        self.assertEqual(pos, Position(1 << 63, 1))
        
    def test_str(self):
        self.assertEqual(str(Position.start()), '---------------------------OX------XO--------------------------- X')

    def test_lt(self):
        self.assertTrue(Position(0, 0) < Position(0, 1))

    def test_empties(self):
        self.assertEqual(Position(0, 0).empties(), 0xFFFFFFFFFFFFFFFF)

    def test_empty_count(self):
        self.assertEqual(Position(0, 0).empty_count(), 64)

    #def test_flipped_codiagonal(self):
    #    self.assertEqual(flipped_codiagonal(Position(0x8080808000000000, 0x4040404000000000)),
    #                    Position(0x000000000000000F, 0x0000000000000F00))

    #def test_flipped_diagonal(self):
    #    self.assertEqual(flipped_diagonal(Position(0x8080808000000000, 0x4040404000000000)),
    #                    Position(0xF000000000000000, 0x00F0000000000000))

    #def test_flipped_horizontal(self):
    #    self.assertEqual(flipped_horizontal(Position(0x8080808000000000, 0x4040404000000000)),
    #                    Position(0x0101010100000000, 0x0202020200000000))

    #def test_flipped_vertical(self):
    #    self.assertEqual(flipped_vertical(Position(0x8080808000000000, 0x4040404000000000)),
    #                    Position(0x0000000080808080, 0x0000000040404040))

    #def test_flipped_to_unique(self):
    #    pos1 = Position(0x8080808000000000, 0x4040404000000000)
    #    pos2 = flipped_vertical(pos1)

    #    self.assertEqual(
    #        flipped_to_unique(pos1), 
    #        flipped_to_unique(pos2))


class IsPosition(unittest.TestCase):
    def test_X_player(self):
        self.assertTrue(is_position('O--------------------------------------------------------------X X'))

    def test_O_player(self):
        self.assertTrue(is_position('O--------------------------------------------------------------X O'))

    def test_no_player(self):
        self.assertFalse(is_position('O--------------------------------------------------------------X'))

    def test_score(self):
        self.assertFalse(is_position('O--------------------------------------------------------------X X % +60'))


class PlayTest(unittest.TestCase):
    def test_play(self):
        pos = play(Position.start(), 19)
        self.assertEqual(pos, Position(0x0000001000000000, 0x0000000818080000))

    def test_play_pass(self):
        start = Position.start()
        passed = play_pass(start)
        passed2 = play_pass(passed)
        self.assertNotEqual(start, passed)
        self.assertNotEqual(passed, passed2)
        self.assertEqual(start, passed2)

    def test_possible_moves(self):
        self.assertEqual(possible_moves(Position.start()), Moves(0x0000102004080000))


#class ChildrenTest(unittest.TestCase):
#    def test_Children_of_start(self):
#        start = Position.start()
#        self.assertEqual(sum(1 for _ in Children(start, 0)), 1)
#        self.assertEqual(sum(1 for _ in Children(start, 1)), 4)
#        self.assertEqual(sum(1 for _ in Children(start, 2)), 12)
#        self.assertEqual(sum(1 for _ in Children(start, 3)), 56)
#        self.assertEqual(sum(1 for _ in Children(start, 4)), 244)
#        self.assertEqual(sum(1 for _ in Children(start, 5)), 1396)
#        self.assertEqual(sum(1 for _ in Children(start, 6)), 8200)


#class AllUniqueChildrenTest(unittest.TestCase):
#    def test_AllUniqueChildren_of_start(self):
#        start = Position.start()
#        self.assertEqual(len(AllUniqueChildren(start, 0)), 1)
#        self.assertEqual(len(AllUniqueChildren(start, 1)), 1)
#        self.assertEqual(len(AllUniqueChildren(start, 2)), 3)
#        self.assertEqual(len(AllUniqueChildren(start, 3)), 14)
#        self.assertEqual(len(AllUniqueChildren(start, 4)), 60)
#        self.assertEqual(len(AllUniqueChildren(start, 5)), 322)
#        self.assertEqual(len(AllUniqueChildren(start, 6)), 1773)


if __name__ == '__main__':
    unittest.main(verbosity=2)
