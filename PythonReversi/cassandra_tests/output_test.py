import unittest
from core import parse_field
from cassandra import Line

class CassandraOutputTest(unittest.TestCase):
    def test_no_selectivity(self):
                   #         #| depth | eval|score|     time [s] |      nodes     |     N/s     | PV
                   # ---------+-------+-----+-----+--------------+----------------+-------------+----
        line = Line("    1'000|14     | +00 |     |        0.020 |             11 |   1'250'000 | PS")

        self.assertEqual(line.index, 1_000)
        self.assertEqual(line.depth, 14)
        self.assertEqual(line.confidence, float('inf'))
        self.assertEqual(line.score, 0)
        self.assertEqual(line.time, '0.020')
        self.assertEqual(line.nodes, 11)
        self.assertEqual(line.speed, 1_250_000)
        self.assertEqual(line.pv, [parse_field('PS')])

    def test_depth_selectivity(self):
                   #         #| depth | eval|score|     time [s] |      nodes     |     N/s     | PV
                   # ---------+-------+-----+-----+--------------+----------------+-------------+----
        line = Line("    1'000|14@1.1 | +00 |     |        0.020 |             11 |   1'250'000 | PS")

        self.assertEqual(line.index, 1_000)
        self.assertEqual(line.depth, 14)
        self.assertEqual(line.confidence, 1.1)
        self.assertEqual(line.score, 0)
        self.assertEqual(line.time, '0.020')
        self.assertEqual(line.nodes, 11)
        self.assertEqual(line.speed, 1_250_000)
        self.assertEqual(line.pv, [parse_field('PS')])

if __name__ == '__main__':
    unittest.main(verbosity=2)
