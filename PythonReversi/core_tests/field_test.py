import unittest
from core import field_to_string, parse_field


class FieldTest(unittest.TestCase):
    def test_field_to_string(self):
        self.assertEqual(field_to_string(0), 'H8')
        self.assertEqual(field_to_string(63), 'A1')
        self.assertEqual(field_to_string(64), 'PS')

    def test_parse_field(self):
        self.assertEqual(parse_field('H8'), 0)
        self.assertEqual(parse_field('A1'), 63)
        self.assertEqual(parse_field('PS'), 64)

    def test_parse_field_lower_case(self):
        self.assertEqual(parse_field('h8'), 0)
        self.assertEqual(parse_field('a1'), 63)
        self.assertEqual(parse_field('ps'), 64)

        
if __name__ == '__main__':
    unittest.main(verbosity=2)
