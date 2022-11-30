import unittest
from core import parse_field
from edax import Line

class EdaxOutputTest(unittest.TestCase):
    def test_exact_depth(self):
        input = '  7|   24   -08        0:00.234      63133975  269803312 b3 C1 b1 A3 b2 H3 a5'

        line = Line(input)

        self.assertEqual(line.index, 7)
        self.assertEqual(line.depth, 24)
        self.assertEqual(line.selectivity, None)
        self.assertEqual(line.confidence, float('inf'))
        self.assertEqual(line.score, -8)
        self.assertEqual(line.time, '0:00.234')
        self.assertEqual(line.nodes, 63133975)
        self.assertEqual(line.speed, 269803312)
        self.assertEqual(line.pv, [parse_field('B3'), parse_field('C1'), parse_field('B1'), parse_field('A3'), parse_field('B2'), parse_field('H3'), parse_field('A5')])

    def test_depth_selectivity(self):
        input = '  8|25@98%  +03        0:00.094       9940593  105750989 G2 b8 B7 a2 A5 b2 G3'

        line = Line(input)

        self.assertEqual(line.index, 8)
        self.assertEqual(line.depth, 25)
        self.assertEqual(line.selectivity, 98)
        self.assertEqual(line.confidence, 2.6)
        self.assertEqual(line.score, +3)
        self.assertEqual(line.time, '0:00.094')
        self.assertEqual(line.nodes, 9940593)
        self.assertEqual(line.speed, 105750989)
        self.assertEqual(line.pv, [parse_field('G2'), parse_field('B8'), parse_field('B7'), parse_field('A2'), parse_field('A5'), parse_field('B2'), parse_field('G3')])

    def test_no_speed(self):
        input = '  1|   14   +18        0:00.000         95959            g8 H7 a8 A6 a4 A7 b6'

        line = Line(input)

        self.assertEqual(line.index, 1)
        self.assertEqual(line.depth, 14)
        self.assertEqual(line.selectivity, None)
        self.assertEqual(line.confidence, float('inf'))
        self.assertEqual(line.score, +18)
        self.assertEqual(line.time, '0:00.000')
        self.assertEqual(line.nodes, 95959)
        self.assertEqual(line.speed, None)
        self.assertEqual(line.pv, [parse_field('G8'), parse_field('H7'), parse_field('A8'), parse_field('A6'), parse_field('A4'), parse_field('A7'), parse_field('B6')])

    def test_pass(self):
        input = '  7|   24   -08        0:00.234      63133975  269803312 ps'

        line = Line(input)

        self.assertEqual(line.index, 7)
        self.assertEqual(line.depth, 24)
        self.assertEqual(line.selectivity, None)
        self.assertEqual(line.confidence, float('inf'))
        self.assertEqual(line.score, -8)
        self.assertEqual(line.time, '0:00.234')
        self.assertEqual(line.nodes, 63133975)
        self.assertEqual(line.speed, 269803312)
        self.assertEqual(line.pv, [parse_field('ps')])

if __name__ == '__main__':
    unittest.main(verbosity=2)
