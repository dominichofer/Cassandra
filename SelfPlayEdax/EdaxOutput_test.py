import unittest
from EdaxOutput import *


class ParseMoveTest(unittest.TestCase):
    def test_Fields(self):
        self.assertEqual(ParseMove('A1'), 63)
        self.assertEqual(ParseMove('B1'), 62)
        self.assertEqual(ParseMove('C1'), 61)
        self.assertEqual(ParseMove('D1'), 60)
        self.assertEqual(ParseMove('E1'), 59)
        self.assertEqual(ParseMove('F1'), 58)
        self.assertEqual(ParseMove('G1'), 57)
        self.assertEqual(ParseMove('H1'), 56)
        self.assertEqual(ParseMove('A2'), 55)
        self.assertEqual(ParseMove('B2'), 54)
        self.assertEqual(ParseMove('C2'), 53)
        self.assertEqual(ParseMove('D2'), 52)
        self.assertEqual(ParseMove('E2'), 51)
        self.assertEqual(ParseMove('F2'), 50)
        self.assertEqual(ParseMove('G2'), 49)
        self.assertEqual(ParseMove('H2'), 48)
        self.assertEqual(ParseMove('A3'), 47)
        self.assertEqual(ParseMove('B3'), 46)
        self.assertEqual(ParseMove('C3'), 45)
        self.assertEqual(ParseMove('D3'), 44)
        self.assertEqual(ParseMove('E3'), 43)
        self.assertEqual(ParseMove('F3'), 42)
        self.assertEqual(ParseMove('G3'), 41)
        self.assertEqual(ParseMove('H3'), 40)
        self.assertEqual(ParseMove('A4'), 39)
        self.assertEqual(ParseMove('B4'), 38)
        self.assertEqual(ParseMove('C4'), 37)
        self.assertEqual(ParseMove('D4'), 36)
        self.assertEqual(ParseMove('E4'), 35)
        self.assertEqual(ParseMove('F4'), 34)
        self.assertEqual(ParseMove('G4'), 33)
        self.assertEqual(ParseMove('H4'), 32)
        self.assertEqual(ParseMove('A5'), 31)
        self.assertEqual(ParseMove('B5'), 30)
        self.assertEqual(ParseMove('C5'), 29)
        self.assertEqual(ParseMove('D5'), 28)
        self.assertEqual(ParseMove('E5'), 27)
        self.assertEqual(ParseMove('F5'), 26)
        self.assertEqual(ParseMove('G5'), 25)
        self.assertEqual(ParseMove('H5'), 24)
        self.assertEqual(ParseMove('A6'), 23)
        self.assertEqual(ParseMove('B6'), 22)
        self.assertEqual(ParseMove('C6'), 21)
        self.assertEqual(ParseMove('D6'), 20)
        self.assertEqual(ParseMove('E6'), 19)
        self.assertEqual(ParseMove('F6'), 18)
        self.assertEqual(ParseMove('G6'), 17)
        self.assertEqual(ParseMove('H6'), 16)
        self.assertEqual(ParseMove('A7'), 15)
        self.assertEqual(ParseMove('B7'), 14)
        self.assertEqual(ParseMove('C7'), 13)
        self.assertEqual(ParseMove('D7'), 12)
        self.assertEqual(ParseMove('E7'), 11)
        self.assertEqual(ParseMove('F7'), 10)
        self.assertEqual(ParseMove('G7'), 9)
        self.assertEqual(ParseMove('H7'), 8)
        self.assertEqual(ParseMove('A8'), 7)
        self.assertEqual(ParseMove('B8'), 6)
        self.assertEqual(ParseMove('C8'), 5)
        self.assertEqual(ParseMove('D8'), 4)
        self.assertEqual(ParseMove('E8'), 3)
        self.assertEqual(ParseMove('F8'), 2)
        self.assertEqual(ParseMove('G8'), 1)
        self.assertEqual(ParseMove('H8'), 0)

    def test_Pass(self):
        self.assertEqual(ParseMove('ps'), 64)
        self.assertEqual(ParseMove('PS'), 64)

    def test_lowercase(self):
        self.assertEqual(ParseMove('a1'), 63)


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
        self.assertEqual(line.pv, [ParseMove('B3'), ParseMove('C1'), ParseMove('B1'), ParseMove('A3'), ParseMove('B2'), ParseMove('H3'), ParseMove('A5')])

    def test_depth_selectivity(self):
        input = '  8|25@98%  +04        0:00.094       9940593  105750989 G2 b8 B7 a2 A5 b2 G3'

        line = Line(input)

        self.assertEqual(line.index, 8)
        self.assertEqual(line.depth, 25)
        self.assertEqual(line.selectivity, 98)
        self.assertEqual(line.confidence, 2.6)
        self.assertEqual(line.score, +4)
        self.assertEqual(line.time, '0:00.094')
        self.assertEqual(line.nodes, 9940593)
        self.assertEqual(line.speed, 105750989)
        self.assertEqual(line.pv, [ParseMove('G2'), ParseMove('B8'), ParseMove('B7'), ParseMove('A2'), ParseMove('A5'), ParseMove('B2'), ParseMove('G3')])

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
        self.assertEqual(line.pv, [ParseMove('G8'), ParseMove('H7'), ParseMove('A8'), ParseMove('A6'), ParseMove('A4'), ParseMove('A7'), ParseMove('B6')])

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
        self.assertEqual(line.pv, [ParseMove('PS')])

if __name__ == '__main__':
    unittest.main()