import unittest
import EdaxScript
from Position import Position

class ParseLine(unittest.TestCase):

    def test_X_to_play(self):
        input = 'XXOO-----------------------------------------------------------X X'

        pos = EdaxScript.ParseLine(input)
        
        self.assertEqual(pos, Position(0xC000000000000001, 0x3000000000000000))
        
    def test_O_to_play(self):
        input = 'XXOO-----------------------------------------------------------X O'

        pos = EdaxScript.ParseLine(input)
        
        self.assertEqual(pos, Position(0x3000000000000000, 0xC000000000000001))
        

class ParseFile(unittest.TestCase):

    def test_X_to_play(self):
        input = ('XXOO-----------------------------------------------------------X X\n'
                 'XXOO-----------------------------------------------------------X O\n')

        pos = EdaxScript.ParseFile(input)
        
        self.assertEqual(pos, [
            Position(0xC000000000000001, 0x3000000000000000),
            Position(0x3000000000000000, 0xC000000000000001)])


if __name__ == '__main__':
    unittest.main()
