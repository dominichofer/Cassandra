import unittest
import EdaxScript
from Position import Position

class ParseLine(unittest.TestCase):
    def test_X_to_play(self):
        input = 'XXOO-----------------------------------------------------------X X'

        ps = EdaxScript.ParseLine(input)
        
        self.assertEqual(ps, Position(0xC000000000000001, 0x3000000000000000))
        
    def test_O_to_play(self):
        input = 'XXOO-----------------------------------------------------------X O'

        ps = EdaxScript.ParseLine(input)
        
        self.assertEqual(ps, Position(0x3000000000000000, 0xC000000000000001))

    def test_score(self):
        input = 'XXOO-----------------------------------------------------------X X % +32'

        ps = EdaxScript.ParseLine(input)
        
        self.assertEqual(ps.score, +32)


if __name__ == '__main__':
    unittest.main()
