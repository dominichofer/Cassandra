import unittest
from core import parse_field
from cassandra import Line

class CassandraOutputTest(unittest.TestCase):
    def test_exact_depth(self):
        line = Line("  1|14      +09           0.020          617'061    30'853'050 G8")

        self.assertEqual(line.index, 1)
        self.assertEqual(line.depth, 14)
        self.assertEqual(line.confidence, float('inf'))
        self.assertEqual(line.score, +9)
        self.assertEqual(line.time, '0.020')
        self.assertEqual(line.nodes, 617_061)
        self.assertEqual(line.speed, 30_853_050)
        self.assertEqual(line.pv, [parse_field('G8')])

    def test_depth_selectivity(self):
        line = Line("  1|14@1.1  +09           0.020          617'061    30'853'050 G8")
        
        self.assertEqual(line.index, 1)
        self.assertEqual(line.depth, 14)
        self.assertEqual(line.confidence, 1.1)
        self.assertEqual(line.score, +9)
        self.assertEqual(line.time, '0.020')
        self.assertEqual(line.nodes, 617_061)
        self.assertEqual(line.speed, 30_853_050)
        self.assertEqual(line.pv, [parse_field('G8')])

    def test_no_speed(self):
        line = Line("  1|14      +09           0.020          617'061               G8")
        
        self.assertEqual(line.index, 1)
        self.assertEqual(line.depth, 14)
        self.assertEqual(line.confidence, float('inf'))
        self.assertEqual(line.score, +9)
        self.assertEqual(line.time, '0.020')
        self.assertEqual(line.nodes, 617_061)
        self.assertEqual(line.speed, None)
        self.assertEqual(line.pv, [parse_field('G8')])

    def test_pass(self):
        line = Line("  1|14      +09           0.020          617'061    30'853'050 PS")
        
        self.assertEqual(line.index, 1)
        self.assertEqual(line.depth, 14)
        self.assertEqual(line.confidence, float('inf'))
        self.assertEqual(line.score, +9)
        self.assertEqual(line.time, '0.020')
        self.assertEqual(line.nodes, 617_061)
        self.assertEqual(line.speed, 30_853_050)
        self.assertEqual(line.pv, [parse_field('ps')])

if __name__ == '__main__':
    unittest.main(verbosity=2)
